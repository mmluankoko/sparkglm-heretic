#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Record and validate real OpenAI API responses; never manage the service.

Examples (run from implementation/):
  python3 scripts/validate_dual_api.py --out-dir evidence/dual-api-quick
  python3 scripts/validate_dual_api.py --stock --out-dir evidence/stock-api
  python3 scripts/validate_dual_api.py --long --out-dir evidence/dual-api-long

--stock discovers the sole advertised model unless --base-model is supplied.
--long adds >1024-token prefill probes and increases the decode budget. Actual
usage, rather than an estimate from characters, gates prefill/decode coverage.
Passing this harness does not prove adapter installation or GPU mathematics.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return type(value) is int and value >= 0


def strict_json(raw):
    def reject_constant(value):
        raise ValueError("non-finite JSON value: " + value)
    return json.loads(raw, parse_constant=reject_constant)


def validate_usage(usage, request, minimum=1, prefill_threshold=0):
    require(isinstance(usage, dict), "missing usage object")
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        require(integer(usage.get(field)), "invalid usage." + field)
    require(usage["prompt_tokens"] > prefill_threshold,
            "prompt_tokens must exceed " + str(prefill_threshold))
    require(minimum <= usage["completion_tokens"] <= request["max_tokens"],
            "completion_tokens outside required decode/request budget")
    require(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
            "usage token totals disagree")
    for field in ("time_to_first_token_ms", "response_token/s"):
        if field in usage:
            value = usage[field]
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                    "invalid timing in usage." + field)
    return usage


def validate_envelope(value, model, expected_object):
    require(isinstance(value, dict), "response is not an object")
    require("error" not in value, "API error: " + repr(value.get("error")))
    require(value.get("object") == expected_object, "unexpected response.object")
    require(isinstance(value.get("id"), str) and bool(value["id"]), "missing response.id")
    require(integer(value.get("created")), "invalid response.created")
    require(value.get("model") == model,
            "response.model mismatch: requested %r, received %r" % (model, value.get("model")))


def validate_math(content):
    require(re.fullmatch(r"\s*(?:\*\*)?42(?:\*\*)?[.!]?\s*", content) is not None,
            "simple math answer must be 42; got " + repr(content))


def validate_blocking(value, request, kind, minimum=1, prefill_threshold=0):
    is_chat = kind == "chat"
    validate_envelope(value, request["model"], "chat.completion" if is_chat else "text_completion")
    choices = value.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, "expected exactly one choice")
    choice = choices[0]
    require(isinstance(choice, dict) and type(choice.get("index")) is int and choice["index"] == 0,
            "expected choice index 0")
    require(choice.get("finish_reason") in ("stop", "length"), "unexpected finish_reason")
    if is_chat:
        message = choice.get("message")
        require(isinstance(message, dict) and message.get("role") == "assistant", "invalid assistant message")
        content = message.get("content")
        reasoning = message.get("reasoning_content") or ""
        require(content is None or isinstance(content, str), "invalid message.content")
        require(isinstance(reasoning, str), "invalid message.reasoning_content")
        content = content or ""
    else:
        content, reasoning = choice.get("text"), ""
        require(isinstance(content, str), "invalid completion text")
    require(bool((content + reasoning).strip()), "empty generated output")
    usage = validate_usage(value.get("usage"), request, minimum, prefill_threshold)
    return {"content": content, "reasoning_content": reasoning, "usage": usage,
            "finish_reason": choice["finish_reason"]}


def parse_sse(raw):
    """Parse SSE bytes, preserving multi-line data and original byte offsets."""
    events, data, event_type, offset = [], [], "message", 0
    for raw_line in raw.splitlines(keepends=True):
        offset += len(raw_line)
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                events.append({"event": event_type, "data": "\n".join(data), "end_offset": offset})
            data, event_type = [], "message"
        elif not line.startswith(":"):
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "data":
                data.append(value)
            elif field == "event":
                event_type = value
    require(not data, "unterminated SSE event at EOF")
    return events


def validate_stream(events, request, minimum=1, prefill_threshold=0):
    content, reasoning, usages, finish_reasons, ids = [], [], [], [], set()
    done, role_seen, first_content_offset = False, False, None
    for event in events:
        require(not done, "SSE data follows [DONE]")
        require(event["event"] != "error", "SSE error event: " + event["data"])
        if event["data"] == "[DONE]":
            done = True
            continue
        value = strict_json(event["data"])
        validate_envelope(value, request["model"], "chat.completion.chunk")
        ids.add(value["id"])
        choices = value.get("choices")
        require(isinstance(choices, list) and len(choices) <= 1, "invalid stream choices")
        if value.get("usage") is not None:
            usages.append(validate_usage(value["usage"], request, minimum, prefill_threshold))
        if not choices:
            require(value.get("usage") is not None, "empty stream choices without usage")
            continue
        choice = choices[0]
        require(isinstance(choice, dict) and type(choice.get("index")) is int and choice["index"] == 0,
                "invalid stream choice index")
        delta = choice.get("delta")
        require(isinstance(delta, dict), "missing stream delta")
        if delta.get("role") is not None:
            require(delta["role"] == "assistant", "invalid stream role")
            role_seen = True
        for field, target in (("content", content), ("reasoning_content", reasoning)):
            text = delta.get(field)
            require(text is None or isinstance(text, str), "invalid stream delta." + field)
            if text:
                require(not finish_reasons, "text follows terminal finish_reason")
                target.append(text)
                if first_content_offset is None:
                    first_content_offset = event["end_offset"]
        if choice.get("finish_reason") is not None:
            require(choice["finish_reason"] in ("stop", "length"), "unexpected stream finish_reason")
            finish_reasons.append(choice["finish_reason"])
    require(done, "SSE missing [DONE]")
    require(role_seen, "SSE missing assistant role")
    require(len(ids) == 1, "SSE response id changed or no chunks received")
    require(len(finish_reasons) == 1, "SSE needs exactly one terminal finish_reason")
    require(bool(usages), "SSE missing requested usage")
    require(all(usage == usages[-1] for usage in usages), "SSE usage chunks disagree")
    require(bool(("".join(content) + "".join(reasoning)).strip()), "empty SSE output")
    return {"content": "".join(content), "reasoning_content": "".join(reasoning),
            "usage": usages[-1], "finish_reason": finish_reasons[0],
            "first_content_end_offset": first_content_offset, "event_count": len(events)}


def validate_models(value, stock, configured_base, adapter):
    require(isinstance(value, dict) and value.get("object") == "list", "invalid models envelope")
    entries = value.get("data")
    require(isinstance(entries, list) and bool(entries), "empty or invalid models list")
    for entry in entries:
        require(isinstance(entry, dict) and entry.get("object") == "model"
                and isinstance(entry.get("id"), str) and bool(entry["id"]), "invalid model entry")
    names = [entry["id"] for entry in entries]
    require(len(names) == len(set(names)), "duplicate advertised model names")
    if stock:
        require(len(names) == 1, "stock baseline requires exactly one advertised base model")
        base = configured_base or names[0]
    else:
        base = configured_base or "glm53-base"
        require(adapter != base, "base and adapter model names must differ")
        require(adapter in names, "adapter model absent from /v1/models")
    require(base in names, "base model absent from /v1/models")
    return {"advertised_models": names, "base_model": base}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class EvidenceClient:
    """All HTTP and evidence I/O goes through this client."""

    def __init__(self, url, key, out_dir, timeout):
        self.url, self.key, self.out_dir, self.timeout = url, key, out_dir, timeout
        # Local and private-cluster probes must reach the requested service,
        # rather than an ambient HTTP proxy that can fabricate failures.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, name, endpoint, payload=None):
        target = self.out_dir / name
        target.mkdir()
        request_data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        write_json(target / "request.json", {"method": "GET" if payload is None else "POST",
                   "url": self.url + endpoint, "body": payload, "authorization": "Bearer [REDACTED]"})
        if request_data is not None:
            (target / "request-body.json").write_bytes(request_data)
        record = {"name": name, "endpoint": endpoint, "started_at": utc_now(),
                  "status": None, "headers": [], "read_timings": [], "error": None}
        started = time.monotonic()
        response = None
        try:
            headers = {"Authorization": "Bearer " + self.key, "Accept-Encoding": "identity"}
            headers["Accept"] = "text/event-stream" if payload and payload.get("stream") else "application/json"
            if request_data is not None:
                headers["Content-Type"] = "application/json"
            request = urllib.request.Request(self.url + endpoint, request_data, headers)
            try:
                response = self.opener.open(request, timeout=self.timeout)
            except urllib.error.HTTPError as error:
                response = error
            record["headers_seconds"] = time.monotonic() - started
            record["status"] = response.status
            record["headers"] = list(response.headers.items())
            record["content_type"] = response.headers.get("Content-Type", "")
            received = 0
            with (target / "response.raw").open("wb") as output:
                while True:
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    output.write(chunk)
                    output.flush()
                    received += len(chunk)
                    record["read_timings"].append({"received_bytes": received,
                                                   "seconds": time.monotonic() - started})
            expected_length = response.headers.get("Content-Length")
            if expected_length is not None:
                require(int(expected_length) == received, "response Content-Length differs from received bytes")
        except Exception as error:
            record["error"] = type(error).__name__ + ": " + str(error)
        except KeyboardInterrupt:
            record["error"] = "KeyboardInterrupt: partial response retained"
            raise
        finally:
            if response is not None:
                response.close()
            (target / "response.raw").touch(exist_ok=True)
            record["elapsed_seconds"] = time.monotonic() - started
            raw = (target / "response.raw").read_bytes()
            record["response_bytes"] = len(raw)
            record["response_sha256"] = hashlib.sha256(raw).hexdigest()
            record["first_byte_seconds"] = (record["read_timings"][0]["seconds"]
                                             if record["read_timings"] else None)
            write_json(target / "transport.json", record)
        return target, record, raw


def generation_cases(args, base):
    sequence = [base, base, base] if args.stock else [base, args.adapter_model, base, args.adapter_model, base]
    short_sequence = sequence if args.long else sequence[:3]
    math_prompt = "What is 17 + 25? Reply with the decimal integer only."
    decode_prompt = ("List every integer from 1 to 1000 in ascending order, separated by commas. "
                     "Do not skip numbers, use an ellipsis, explain, or stop before 1000.")
    families = [("math-chat", "chat", False, math_prompt, 32, 1, True, 0),
                ("math-sse", "chat", True, math_prompt, 32, 1, True, 0),
                ("math-completions", "completion", False,
                 "Question: What is 17 + 25?\nAnswer (decimal integer only):", 32, 1, True, 0),
                ("decode-chat", "chat", False, decode_prompt, args.decode_tokens, args.min_decode_tokens, False, 0),
                ("decode-sse", "chat", True, decode_prompt, args.decode_tokens, args.min_decode_tokens, False, 0)]
    if args.long or args.long_prefill:
        context = "\n".join("Record %04d: amber birch cedar delta elm fir granite hazel." % i
                            for i in range(args.prefill_lines))
        prompt = "Read these records.\n" + context + "\nNow answer: " + math_prompt
        families.append(("prefill-chat", "chat", False, prompt, 32, 1, True, args.chunk_threshold))
    for family, kind, stream, prompt, budget, minimum, check_math, threshold in families:
        models = short_sequence if family.startswith(("decode", "prefill")) else sequence
        for turn, model in enumerate(models, 1):
            payload = {"model": model, "temperature": 0, "seed": args.seed,
                       "max_tokens": budget, "stream": stream, "top_p": 1,
                       "repetition_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0}
            if kind == "chat":
                payload["messages"] = [{"role": "user", "content": prompt}]
                payload["enable_thinking"] = False
                payload["chat_template_kwargs"] = {"enable_thinking": False}
                payload["min_tokens"] = minimum
            else:
                payload["prompt"], payload["stop"] = prompt, ["\n"]
            if stream:
                payload["stream_options"] = {"include_usage": True}
            yield {"name": "%s-%02d-%s" % (family, turn, "base" if model == base else "heretic"),
                   "family": family, "kind": kind, "model": model, "payload": payload,
                   "minimum": minimum, "check_math": check_math, "prefill_threshold": threshold}


def compare_repetitions(results, base, adapter):
    checks = []
    for family in sorted({item["family"] for item in results}):
        entries = [item for item in results if item["family"] == family]
        for model in (base, adapter):
            repeats = [item for item in entries if item["model"] == model and item.get("output")]
            if len(repeats) < 2:
                continue
            reference = repeats[0]["output"]
            signature = lambda value: (value["content"], value["reasoning_content"], value["finish_reason"],
                                       value["usage"]["prompt_tokens"], value["usage"]["completion_tokens"])
            stable = all(signature(item["output"]) == signature(reference) for item in repeats[1:])
            checks.append({"family": family, "model": model, "passed": stable,
                           "check": "base_output_stable_across_switches" if model == base else "adapter_repeat_stable",
                           "requests": [item["name"] for item in repeats],
                           "note": "A mismatch is a reproducibility/possible contamination failure; text alone cannot identify its cause."})
    return checks


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8002", help="service root or /v1 URL")
    parser.add_argument("--key", default=os.environ.get("SPARKGLM_API_KEY", ""), help="API key (or SPARKGLM_API_KEY; omitted from evidence)")
    parser.add_argument("--out-dir", type=Path, required=True, help="new evidence directory; must not exist")
    parser.add_argument("--stock", action="store_true", help="single-model stock baseline; auto-discover model")
    parser.add_argument("--base-model", help="dual default: glm53-base; stock default: sole advertised model")
    parser.add_argument("--adapter-model", default="heretic")
    parser.add_argument("--long", action="store_true", help="1024-token decode and >1024-token prefill probes")
    parser.add_argument("--long-prefill", action="store_true", help="add chunk-sized prefill probes in quick mode")
    parser.add_argument("--decode-tokens", type=int, help="default: 256 quick, 1024 long")
    parser.add_argument("--min-decode-tokens", type=int, help="required actual decode; default: 128 quick, 512 long")
    parser.add_argument("--prefill-lines", type=int, default=256, help="fixed record lines; validated using actual prompt usage")
    parser.add_argument("--chunk-threshold", type=int, default=1024, help="prefill usage must exceed this server chunk threshold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=600, help="HTTP socket timeout in seconds")
    args = parser.parse_args(argv)
    args.url = args.url.rstrip("/")
    if args.url.endswith("/v1"):
        args.url = args.url[:-3]
    url = urllib.parse.urlsplit(args.url)
    if url.scheme not in ("http", "https") or not url.netloc or url.query or url.fragment or url.username:
        parser.error("--url must be an HTTP(S) service URL without credentials, query or fragment")
    args.decode_tokens = args.decode_tokens if args.decode_tokens is not None else (1024 if args.long else 256)
    args.min_decode_tokens = args.min_decode_tokens if args.min_decode_tokens is not None else (512 if args.long else 128)
    if not (0 < args.min_decode_tokens <= args.decode_tokens and args.prefill_lines > 0
            and args.chunk_threshold >= 1024 and 0 <= args.seed < 2**64
            and math.isfinite(args.timeout) and args.timeout > 0):
        parser.error("invalid decode budgets, prefill dimensions, seed or timeout")
    return args


def main(argv=None):
    args = arguments(argv)
    try:
        args.out_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Evidence directory already exists: " + str(args.out_dir), file=sys.stderr)
        return 2
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "key"}
    summary = {"schema": "onlineheretic.dual-api-validation/v1", "started_at": utc_now(),
               "mode": "stock-baseline" if args.stock else "dual-model", "profile": "long" if args.long else "quick",
               "config": config, "results": [], "checks": [], "errors": [], "passed": False,
               "limitations": ["Text equality or divergence does not prove adapter installation or GPU numerical correctness.",
                               "Prefill usage above the configured chunk size is a coverage candidate; worker logs must confirm actual chunk scheduling."]}
    write_json(args.out_dir / "summary.json", summary)
    client = EvidenceClient(args.url, args.key, args.out_dir, args.timeout)
    exit_code = 1
    try:
        directory, transport, raw = client.request("models", "/v1/models")
        require(transport["error"] is None, "models transport failed: " + str(transport["error"]))
        require(transport["status"] == 200, "models HTTP status " + str(transport["status"]))
        require("application/json" in transport["content_type"].lower(), "unexpected models content type")
        models = strict_json(raw)
        write_json(directory / "response.json", models)
        summary["models"] = validate_models(models, args.stock, args.base_model, args.adapter_model)
        base = summary["models"]["base_model"]
        for case in generation_cases(args, base):
            item = {key: case[key] for key in ("name", "family", "kind", "model")}
            item["passed"] = False
            summary["results"].append(item)
            endpoint = "/v1/chat/completions" if case["kind"] == "chat" else "/v1/completions"
            directory, transport, raw = client.request(case["name"], endpoint, case["payload"])
            item["elapsed_seconds"] = transport["elapsed_seconds"]
            try:
                require(transport["error"] is None, "transport failed: " + str(transport["error"]))
                require(transport["status"] == 200, "HTTP status " + str(transport["status"]))
                if case["payload"]["stream"]:
                    require("text/event-stream" in transport["content_type"].lower(), "unexpected SSE content type")
                    events = parse_sse(raw)
                    write_json(directory / "events.json", events)
                    output = validate_stream(events, case["payload"], case["minimum"], case["prefill_threshold"])
                    output["first_content_seconds"] = next((tick["seconds"] for tick in transport["read_timings"]
                        if tick["received_bytes"] >= output["first_content_end_offset"]), None)
                else:
                    require("application/json" in transport["content_type"].lower(), "unexpected JSON content type")
                    value = strict_json(raw)
                    write_json(directory / "response.json", value)
                    output = validate_blocking(value, case["payload"], case["kind"], case["minimum"], case["prefill_threshold"])
                item["output"] = output
                if case["check_math"]:
                    validate_math(output["content"])
                item["passed"] = True
            except Exception as error:
                item["error"] = type(error).__name__ + ": " + str(error)
            write_json(directory / "validation.json", item)
            write_json(args.out_dir / "summary.json", summary)
            print(("PASS " if item["passed"] else "FAIL ") + case["name"], flush=True)
        summary["checks"] = compare_repetitions(summary["results"], base, args.adapter_model)
        summary["cross_model_observations"] = []
        for family in sorted({item["family"] for item in summary["results"]}):
            entries = [item for item in summary["results"] if item["family"] == family and item.get("output")]
            base_outputs = [item["output"] for item in entries if item["model"] == base]
            adapter_outputs = [item["output"] for item in entries if item["model"] == args.adapter_model]
            if base_outputs and adapter_outputs:
                summary["cross_model_observations"].append({"family": family,
                    "same_text": base_outputs[0]["content"] == adapter_outputs[0]["content"],
                    "note": "Informational only; matching outputs are not an adapter failure."})
        summary["passed"] = all(item["passed"] for item in summary["results"]) and all(check["passed"] for check in summary["checks"])
        exit_code = 0 if summary["passed"] else 1
    except KeyboardInterrupt:
        summary["errors"].append("Interrupted by user; partial raw responses retained")
        exit_code = 130
    except Exception as error:
        summary["errors"].append(type(error).__name__ + ": " + str(error))
    finally:
        summary["finished_at"] = utc_now()
        write_json(args.out_dir / "summary.json", summary)
    print(("PASS" if summary["passed"] else "FAIL") + ": " + str(args.out_dir / "summary.json"), flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
