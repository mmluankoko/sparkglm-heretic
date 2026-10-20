#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Lossless, narrow converter for the pinned GLM5.3 Heretic V3 adapter.

Writes a PEFT-style per-expert safetensors package. Does not assert engine
support, merge weights, load a model, or contact the network. Standard library
only. This deliberately rejects other artifacts until reviewed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct

EXPECTED_SHA = '6a1cf5e5e3d50889c7b1533fca8969f944f61293bd14c1a93365feb54b51fdc3'

def read_header(f):
    def scalar(fmt):
        n=struct.calcsize('<'+fmt)
        return struct.unpack('<'+fmt, f.read(n))[0]
    def string():
        return f.read(scalar('Q')).decode('utf-8')
    def value(t):
        formats={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
        if t in formats:return scalar(formats[t])
        if t==8:return string()
        if t==9:
            item_type=scalar('I');count=scalar('Q')
            return [value(item_type) for _ in range(count)]
        raise ValueError(f'Unknown GGUF metadata type {t}')
    assert f.read(4)==b'GGUF'
    assert scalar('I')==3
    nt,nkv=scalar('Q'),scalar('Q')
    metadata={}
    for _ in range(nkv):
        key=string();metadata[key]=value(scalar('I'))
    tensors=[]
    for _ in range(nt):
        name=string();ndim=scalar('I')
        shape=[scalar('Q') for _ in range(ndim)]
        tensors.append({'name':name,'shape':shape,'type':scalar('I'),'offset':scalar('Q')})
    alignment=metadata.get('general.alignment',32)
    start=(f.tell()+alignment-1)//alignment*alignment
    return metadata,tensors,start

def base_headers(directory):
    """Read safetensors JSON headers only; never read base weight payloads."""
    index = directory / 'model.safetensors.index.json'
    if index.is_file():
        names = sorted(set(json.loads(index.read_text())['weight_map'].values()))
        paths = [directory / name for name in names]
    else:
        paths = sorted(directory.glob('*.safetensors'))
    if not paths:
        raise ValueError('No base safetensors shards found')
    result = {}
    for path in paths:
        with path.open('rb') as handle:
            size = struct.unpack('<Q', handle.read(8))[0]
            if size > 100_000_000:
                raise ValueError(f'Unexpected safetensors header size: {path}')
            header = json.loads(handle.read(size))
        result.update({name: value for name, value in header.items() if name != '__metadata__'})
    return result


def convert(source, dest, base_directory):
    SOURCE, DEST = source, dest
    if (DEST / 'adapter_model.safetensors').exists():
        raise FileExistsError('Output adapter already exists; choose an empty output directory')
    source_sha=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert source_sha==EXPECTED_SHA,'Pinned adapter SHA256 mismatch'
    with SOURCE.open('rb') as f:
        metadata,tensors,start=read_header(f)
    assert metadata['adapter.type']=='lora'
    assert metadata['adapter.lora.alpha']==1.0
    base=base_headers(base_directory)
    entries=[]
    modules=set()
    seen=set()
    for tensor in tensors:
        assert tensor['type']==1,'Expected GGML F16'
        match=re.fullmatch(r'blk\.(\d+)\.(attn_output|ffn_down_shexp|ffn_down_exps)\.weight\.lora_([ab])',tensor['name'])
        assert match,tensor['name']
        layer,kind,ab=match.groups()
        dims=tensor['shape']
        expert_count=dims[2] if kind=='ffn_down_exps' else 1
        assert len(dims)==(3 if kind=='ffn_down_exps' else 2)
        assert expert_count==(288 if kind=='ffn_down_exps' else 1)
        shape=list(reversed(dims[:2]))
        assert (shape[0] if ab=='a' else shape[1])==1
        slice_bytes=2*math.prod(shape)
        for expert in range(expert_count):
            suffix={'attn_output':'self_attn.o_proj','ffn_down_shexp':'mlp.shared_experts.down_proj','ffn_down_exps':f'mlp.experts.{expert}.down_proj'}[kind]
            module=f'model.language_model.layers.{layer}.{suffix}'
            w=base[module+'.weight']
            logical=w['shape'][:]
            if w['dtype']=='U8':logical[1]*=2
            assert shape==([1,logical[1]] if ab=='a' else [logical[0],1]),module
            name=f'base_model.model.{module}.lora_{ab.upper()}.weight'
            assert name not in seen;seen.add(name);modules.add(module)
            entries.append({'name':name,'shape':shape,'source_tensor':tensor['name'],'expert':expert if kind=='ffn_down_exps' else None,'source_offset':start+tensor['offset']+expert*slice_bytes,'bytes':slice_bytes})
    assert len(entries)==11040 and len(modules)==5520
    header={'__metadata__':{'format':'pt','source_sha256':source_sha,'conversion':'GGUF F16 to PEFT per-expert F16; payload preserved; engine support unverified'}}
    offset=0
    for entry in entries:
        header[entry['name']]={'dtype':'F16','shape':entry['shape'],'data_offsets':[offset,offset+entry['bytes']]}
        offset+=entry['bytes']
    assert offset==68251648
    raw=json.dumps(header,separators=(',',':')).encode()
    raw+=b' '*((-len(raw))%8)
    DEST.mkdir(parents=True,exist_ok=True)
    output=DEST/'adapter_model.safetensors'
    temp=output.with_suffix('.partial')
    with SOURCE.open('rb') as src,temp.open('wb') as out:
        out.write(struct.pack('<Q',len(raw)));out.write(raw)
        for entry in entries:
            src.seek(entry['source_offset']);data=src.read(entry['bytes'])
            assert len(data)==entry['bytes']
            out.write(data)
    temp.replace(output)
    config={'base_model_name_or_path':'nvidia/GLM-5.3-Flash-NVFP4','revision':'da920bb0b9f4a06727223a349e55468e38352348','peft_type':'LORA','task_type':'CAUSAL_LM','inference_mode':True,'r':1,'lora_alpha':1,'lora_dropout':0.0,'bias':'none','fan_in_fan_out':False,'use_rslora':False,'use_dora':False,'modules_to_save':None,'target_modules':sorted(modules)}
    (DEST/'adapter_config.json').write_text(json.dumps(config,indent=2)+'\n')
    provenance={'original_base_used_for_optimization':'unsloth GLM-5.3-Flash UD-IQ4_XS (per source model card)','declared_deployment_base':'nvidia/GLM-5.3-Flash-NVFP4; retargeted, not evaluated','source_repo':'MorinoNushi/GLM-5.3-Flash-Heretic-Abliterated-LoRA-V3-GGUF','source_revision':'c5623df0c106cb05a6625669011f1336d6460e09','source_sha256':source_sha,'source_metadata':metadata,'conversion':'F16 bytes unchanged; GGML shape order reversed for row-major; expert dimension split into 288 independent A/B pairs','engine_status':'Format conversion only. SparkGLM serving implementation and runtime validation are tracked separately in the sparkglm-heretic repository.','tensor_map':entries}
    (DEST/'conversion_manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')

    # Reopen output and reconstruct every source GGUF tensor independently.
    # Compare whole tensors, not just a few sampled parameter values.
    with output.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0];out_header=json.loads(f.read(n));data_start=8+n
        checks=[]
        with SOURCE.open('rb') as src:
            for t in tensors:
                matching=[e for e in entries if e['source_tensor']==t['name']]
                rebuilt=bytearray()
                for entry in matching:
                    h=out_header[entry['name']]
                    assert h['dtype']=='F16' and h['shape']==entry['shape']
                    lo,hi=h['data_offsets'];f.seek(data_start+lo)
                    rebuilt.extend(f.read(hi-lo))
                src.seek(start+t['offset']);original=src.read(2*math.prod(t['shape']))
                assert rebuilt==original,t['name']
                checks.append({'name':t['name'],'bytes':len(original),'sha256':hashlib.sha256(original).hexdigest()})
        assert max(v['data_offsets'][1] for k,v in out_header.items() if k!='__metadata__')==output.stat().st_size-data_start
    result={'source_sha256_verified':True,'all_134_source_tensors_reconstructed_byte_identically':True,'output_tensor_count':len(entries),'target_matrix_count':len(modules),'payload_bytes':offset,'safetensors_file_bytes':output.stat().st_size,'safetensors_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'config_scaling':1.0,'format_reader_validation':'Independent safetensors reader pending','model_inference_tested':False,'source_tensor_checks':checks}
    (DEST/'conversion_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_tensor_checks'},indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='pinned V3 GGUF LoRA file')
    parser.add_argument('--output', type=Path, required=True, help='output PEFT adapter directory')
    parser.add_argument('--base', type=Path, required=True, help='existing NVIDIA NVFP4 checkpoint directory')
    args = parser.parse_args()
    convert(args.source, args.output, args.base)
