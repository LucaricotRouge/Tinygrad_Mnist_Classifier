
const mnist_mlp = (() => {
const getTensorBuffer = (safetensorBuffer, tensorMetadata) => {
  return safetensorBuffer.subarray(...tensorMetadata.data_offsets);
};

const getTensorMetadata = (safetensorBuffer) => {
    const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
    const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
    return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}]));
};

const createEmptyBuf = (device, size) => {
    return device.createBuffer({size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
};

const createUniformBuf = (device, size) => {
  return device.createBuffer({size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST})
}

const createInfinityUniformBuf = (device) => {
  const size = 4;
  const buf = device.createBuffer({
    mappedAtCreation: true,
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  });
  new Float32Array(buf.getMappedRange())[0] = Infinity;
  buf.unmap();
  return buf;
};

const createWeightBuf = (device, size, data) => {
  const buf = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
  new Uint8Array(buf.getMappedRange()).set(data); buf.unmap();
  return buf;
};

const addComputePass = (device, commandEncoder, pipeline, layout, infinityUniformBuf, bufs, workgroup) => {
  const bindGroup = device.createBindGroup({
    layout: layout,
    entries: [
      { binding: 0, resource: { buffer: infinityUniformBuf } },
      ...bufs.map((buffer, index) => ({ binding: index + 1, resource: { buffer } }))
    ]
  });

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...workgroup);
  passEncoder.end();
};

const r_32_8_4_4_98 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_512:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_784:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_401408:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_512:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var acc1: array<f32,4>;
  var lidx1 = i32(lindex.y); /* 4 */
  var lidx0 = i32(lindex.x); /* 8 */
  var gidx0 = i32(gindex.x); /* 32 */
  var precast0 = lidx0;
  var precast1 = lidx1;
  var precast2 = (bitcast<u32>(precast0)<<2u);
  var precast3 = (bitcast<u32>(precast1)<<5u);
  var cast0 = bitcast<i32>(precast3);
  var alu0 = (bitcast<i32>(precast2)+cast0);
  var precast4 = gidx0;
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc1[0] = 0.0f;
  acc1[1] = 0.0f;
  acc1[2] = 0.0f;
  acc1[3] = 0.0f;
  var precast5 = (bitcast<u32>(precast4)<<4u);
  var alu9 = (lidx1+bitcast<i32>(precast5));
  var val0 = data3_512[alu9];
  var alu10 = (alu9+4);
  var val1 = data3_512[alu10];
  var alu11 = (alu9+8);
  var val2 = data3_512[alu11];
  var alu12 = (alu9+12);
  var val3 = data3_512[alu12];
  for (var Ridx0 = 0; Ridx0 < 98; Ridx0++) {
    var precast6 = Ridx0;
    var precast7 = (bitcast<u32>(precast6)<<3u);
    var alu13 = (lidx0+bitcast<i32>(precast7));
    var val4 = data1_784[alu13];
    var alu14 = (alu13+(gidx0*12544)+(lidx1*784));
    var val5 = data2_401408[alu14];
    var val6 = data2_401408[(alu14+3136)];
    var val7 = data2_401408[(alu14+6272)];
    var val8 = data2_401408[(alu14+9408)];
    acc0[0] = (acc0[0]+(val4*val5));
    acc0[1] = (acc0[1]+(val4*val6));
    acc0[2] = (acc0[2]+(val4*val7));
    acc0[3] = (acc0[3]+(val4*val8));
  }
  temp0[alu0] = acc0[0];
  temp0[(alu0+1)] = acc0[1];
  temp0[(alu0+2)] = acc0[2];
  temp0[(alu0+3)] = acc0[3];
  workgroupBarrier();
  for (var Ridx102 = 0; Ridx102 < 8; Ridx102++) {
    var precast8 = Ridx102;
    var precast9 = (bitcast<u32>(precast8)<<2u);
    var alu25 = (cast0+bitcast<i32>(precast9));
    var val9 = temp0[alu25];
    var val10 = temp0[(alu25+1)];
    var val11 = temp0[(alu25+2)];
    var val12 = temp0[(alu25+3)];
    acc1[0] = (acc1[0]+val9);
    acc1[1] = (acc1[1]+val10);
    acc1[2] = (acc1[2]+val11);
    acc1[3] = (acc1[3]+val12);
  }
  var alu31 = (acc1[0]+val0);
  var alu32 = (acc1[1]+val1);
  var alu33 = (acc1[2]+val2);
  var alu34 = (acc1[3]+val3);
  var alu35 = ((bool(lidx0))!=true);
  if (alu35) {
    data0_512[alu9] = (alu31*(1/(1.0f+exp2((alu31*-1.4426950408889634f)))));
  }
  if (alu35) {
    data0_512[alu10] = (alu32*(1/(1.0f+exp2((alu32*-1.4426950408889634f)))));
  }
  if (alu35) {
    data0_512[alu11] = (alu33*(1/(1.0f+exp2((alu33*-1.4426950408889634f)))));
  }
  if (alu35) {
    data0_512[alu12] = (alu34*(1/(1.0f+exp2((alu34*-1.4426950408889634f)))));
  }
}`;

const r_32_8_4_4_64 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_512:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_512:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_262144:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_512:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var acc1: array<f32,4>;
  var lidx1 = i32(lindex.y); /* 4 */
  var lidx0 = i32(lindex.x); /* 8 */
  var gidx0 = i32(gindex.x); /* 32 */
  var precast0 = lidx0;
  var precast1 = lidx1;
  var cast0 = bitcast<u32>(precast1);
  var precast2 = (bitcast<u32>(precast0)<<2u);
  var precast3 = (cast0<<5u);
  var cast1 = bitcast<i32>(precast3);
  var alu0 = (bitcast<i32>(precast2)+cast1);
  var precast4 = gidx0;
  var cast2 = bitcast<u32>(precast4);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc1[0] = 0.0f;
  acc1[1] = 0.0f;
  acc1[2] = 0.0f;
  acc1[3] = 0.0f;
  var precast5 = (cast2<<4u);
  var alu9 = (lidx1+bitcast<i32>(precast5));
  var val0 = data3_512[alu9];
  var alu10 = (alu9+4);
  var val1 = data3_512[alu10];
  var alu11 = (alu9+8);
  var val2 = data3_512[alu11];
  var alu12 = (alu9+12);
  var val3 = data3_512[alu12];
  var precast6 = (cast2<<13u);
  var precast7 = (cast0<<9u);
  for (var Ridx0 = 0; Ridx0 < 64; Ridx0++) {
    var precast8 = Ridx0;
    var precast9 = (bitcast<u32>(precast8)<<3u);
    var alu13 = (lidx0+bitcast<i32>(precast9));
    var val4 = data1_512[alu13];
    var alu14 = (alu13+bitcast<i32>(precast6)+bitcast<i32>(precast7));
    var val5 = data2_262144[alu14];
    var val6 = data2_262144[(alu14+2048)];
    var val7 = data2_262144[(alu14+4096)];
    var val8 = data2_262144[(alu14+6144)];
    acc0[0] = (acc0[0]+(val4*val5));
    acc0[1] = (acc0[1]+(val4*val6));
    acc0[2] = (acc0[2]+(val4*val7));
    acc0[3] = (acc0[3]+(val4*val8));
  }
  temp0[alu0] = acc0[0];
  temp0[(alu0+1)] = acc0[1];
  temp0[(alu0+2)] = acc0[2];
  temp0[(alu0+3)] = acc0[3];
  workgroupBarrier();
  for (var Ridx102 = 0; Ridx102 < 8; Ridx102++) {
    var precast10 = Ridx102;
    var precast11 = (bitcast<u32>(precast10)<<2u);
    var alu25 = (cast1+bitcast<i32>(precast11));
    var val9 = temp0[alu25];
    var val10 = temp0[(alu25+1)];
    var val11 = temp0[(alu25+2)];
    var val12 = temp0[(alu25+3)];
    acc1[0] = (acc1[0]+val9);
    acc1[1] = (acc1[1]+val10);
    acc1[2] = (acc1[2]+val11);
    acc1[3] = (acc1[3]+val12);
  }
  var alu31 = (acc1[0]+val0);
  var alu32 = (acc1[1]+val1);
  var alu33 = (acc1[2]+val2);
  var alu34 = (acc1[3]+val3);
  var alu35 = ((bool(lidx0))!=true);
  if (alu35) {
    data0_512[alu9] = (alu31*(1/(1.0f+exp2((alu31*-1.4426950408889634f)))));
  }
  if (alu35) {
    data0_512[alu10] = (alu32*(1/(1.0f+exp2((alu32*-1.4426950408889634f)))));
  }
  if (alu35) {
    data0_512[alu11] = (alu33*(1/(1.0f+exp2((alu33*-1.4426950408889634f)))));
  }
  if (alu35) {
    data0_512[alu12] = (alu34*(1/(1.0f+exp2((alu34*-1.4426950408889634f)))));
  }
}`;

const r_10_16_32 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_10:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_512:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_5120:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_10:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 10 */
  var lidx0 = i32(lindex.x); /* 16 */
  var precast0 = gidx0;
  var precast1 = lidx0;
  acc0[0] = 0.0f;
  acc1[0] = 0.0f;
  var val0 = data3_10[gidx0];
  var precast2 = (bitcast<u32>(precast0)<<9u);
  var precast3 = (bitcast<u32>(precast1)<<5u);
  for (var Ridx0 = 0; Ridx0 < 32; Ridx0++) {
    var alu2 = (bitcast<i32>(precast3)+Ridx0);
    var val1 = data1_512[alu2];
    var val2 = data2_5120[(alu2+bitcast<i32>(precast2))];
    acc0[0] = (acc0[0]+(val1*val2));
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  for (var Ridx102 = 0; Ridx102 < 16; Ridx102++) {
    var val3 = temp0[Ridx102];
    acc1[0] = (acc1[0]+val3);
  }
  var alu9 = ((bool(lidx0))!=true);
  if (alu9) {
    data0_10[gidx0] = (acc1[0]+val0);
  }
}`;

const setupNet = async (device, safetensor) => {
    const metadata = getTensorMetadata(safetensor);
    const infinityBuf = createInfinityUniformBuf(device);

    const layouts=[device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]})]

    const buf_0 = createEmptyBuf(device, 2048);;
    const input0 = createEmptyBuf(device, 3136);;
    const buf_1 = createWeightBuf(device, 1605632, getTensorBuffer(safetensor, metadata['layers.1.weight']));
    const buf_2 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['layers.1.bias']));
    const buf_3 = createEmptyBuf(device, 2048);;
    const buf_4 = createWeightBuf(device, 1048576, getTensorBuffer(safetensor, metadata['layers.3.weight']));
    const buf_5 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['layers.3.bias']));
    const output0 = createEmptyBuf(device, 40);;
    const buf_6 = createWeightBuf(device, 20480, getTensorBuffer(safetensor, metadata['layers.5.weight']));
    const buf_7 = createWeightBuf(device, 40, getTensorBuffer(safetensor, metadata['layers.5.bias']));

    const gpuWriteBuffer0 = device.createBuffer({size:input0.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });

    const gpuReadBuffer0 = device.createBuffer({size:output0.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const kernels = [r_32_8_4_4_98, r_32_8_4_4_64, r_10_16_32];
    const pipelines = await Promise.all(kernels.map(async (name, i) => {
      return await device.createComputePipelineAsync({
          layout: device.createPipelineLayout({
              bindGroupLayouts: [layouts[i]],
          }),
          compute: {
              module: device.createShaderModule({
                  code: name,
              }),
              entryPoint: "main",
          },
      });
  }))

    return async (_input0) => {
        const commandEncoder = device.createCommandEncoder();
        await gpuWriteBuffer0.mapAsync(GPUMapMode.WRITE);
        new Float32Array(gpuWriteBuffer0.getMappedRange()).set(_input0);
        gpuWriteBuffer0.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer0, 0, input0, 0, gpuWriteBuffer0.size);
        addComputePass(device, commandEncoder, pipelines[0], layouts[0], infinityBuf, [buf_0, input0, buf_1, buf_2], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[1], layouts[1], infinityBuf, [buf_3, buf_0, buf_4, buf_5], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[2], layouts[2], infinityBuf, [output0, buf_3, buf_6, buf_7], [10, 1, 1]);
        commandEncoder.copyBufferToBuffer(output0, 0, gpuReadBuffer0, 0, output0.size);
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        await gpuReadBuffer0.mapAsync(GPUMapMode.READ);
        const resultBuffer0 = new Float32Array(gpuReadBuffer0.size/4);
        resultBuffer0.set(new Float32Array(gpuReadBuffer0.getMappedRange()));
        gpuReadBuffer0.unmap();
        return [resultBuffer0];
    }
}
const load = async (device, weight_path) => { return await fetch(weight_path).then(x => x.arrayBuffer()).then(x => setupNet(device, new Uint8Array(x))); }
return { load, setupNet };
})();
export default mnist_mlp;
