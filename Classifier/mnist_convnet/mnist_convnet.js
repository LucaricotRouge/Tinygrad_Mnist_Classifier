
const mnist_convnet = (() => {
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

const r_4_3_2_8_8_3_4_5_5 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_18432:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_784:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_800:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@compute @workgroup_size(2,8,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,12>;
  var lidx0 = i32(lindex.x); /* 2 */
  var gidx0 = i32(gindex.x); /* 3 */
  var gidx1 = i32(gindex.y); /* 4 */
  var lidx1 = i32(lindex.y); /* 8 */
  var lidx2 = i32(lindex.z); /* 8 */
  var precast0 = gidx1;
  var precast1 = lidx0;
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  var alu12 = (lidx2*3);
  var precast2 = (bitcast<u32>(precast0)<<3u);
  var precast3 = (bitcast<u32>(precast1)<<2u);
  var alu13 = (bitcast<i32>(precast2)+bitcast<i32>(precast3));
  var val0 = data3_32[alu13];
  var val1 = data3_32[(alu13+1)];
  var val2 = data3_32[(alu13+2)];
  var val3 = data3_32[(alu13+3)];
  for (var Ridx0 = 0; Ridx0 < 5; Ridx0++) {
    var alu14 = ((gidx1*200)+(lidx0*100)+(Ridx0*5));
    var val4 = data2_800[alu14];
    var val5 = data2_800[(alu14+1)];
    var val6 = data2_800[(alu14+2)];
    var val7 = data2_800[(alu14+3)];
    var val8 = data2_800[(alu14+4)];
    var val9 = data2_800[(alu14+25)];
    var val10 = data2_800[(alu14+26)];
    var val11 = data2_800[(alu14+27)];
    var val12 = data2_800[(alu14+28)];
    var val13 = data2_800[(alu14+29)];
    var val14 = data2_800[(alu14+50)];
    var val15 = data2_800[(alu14+51)];
    var val16 = data2_800[(alu14+52)];
    var val17 = data2_800[(alu14+53)];
    var val18 = data2_800[(alu14+54)];
    var val19 = data2_800[(alu14+75)];
    var val20 = data2_800[(alu14+76)];
    var val21 = data2_800[(alu14+77)];
    var val22 = data2_800[(alu14+78)];
    var val23 = data2_800[(alu14+79)];
    var alu15 = ((gidx0*224)+(lidx1*28)+(Ridx0*28)+alu12);
    var val24 = data1_784[alu15];
    var val25 = data1_784[(alu15+1)];
    var val26 = data1_784[(alu15+2)];
    var val27 = data1_784[(alu15+3)];
    var val28 = data1_784[(alu15+4)];
    var val29 = data1_784[(alu15+5)];
    var val30 = data1_784[(alu15+6)];
    acc0[4] = (acc0[4]+(val25*val4)+(val26*val5)+(val27*val6)+(val28*val7)+(val29*val8));
    acc0[5] = (acc0[5]+(val25*val9)+(val26*val10)+(val27*val11)+(val28*val12)+(val29*val13));
    acc0[6] = (acc0[6]+(val25*val14)+(val26*val15)+(val27*val16)+(val28*val17)+(val29*val18));
    acc0[7] = (acc0[7]+(val25*val19)+(val26*val20)+(val27*val21)+(val28*val22)+(val29*val23));
    acc0[8] = (acc0[8]+(val26*val4)+(val27*val5)+(val28*val6)+(val29*val7)+(val30*val8));
    acc0[9] = (acc0[9]+(val26*val9)+(val27*val10)+(val28*val11)+(val29*val12)+(val30*val13));
    acc0[10] = (acc0[10]+(val26*val14)+(val27*val15)+(val28*val16)+(val29*val17)+(val30*val18));
    acc0[11] = (acc0[11]+(val26*val19)+(val27*val20)+(val28*val21)+(val29*val22)+(val30*val23));
    acc0[1] = (acc0[1]+(val24*val9)+(val25*val10)+(val26*val11)+(val27*val12)+(val28*val13));
    acc0[2] = (acc0[2]+(val24*val14)+(val25*val15)+(val26*val16)+(val27*val17)+(val28*val18));
    acc0[3] = (acc0[3]+(val24*val19)+(val25*val20)+(val26*val21)+(val27*val22)+(val28*val23));
    acc0[0] = (acc0[0]+(val24*val4)+(val25*val5)+(val26*val6)+(val27*val7)+(val28*val8));
  }
  var alu29 = (acc0[0]+val0);
  var alu30 = (acc0[1]+val1);
  var alu31 = (acc0[2]+val2);
  var alu32 = (acc0[3]+val3);
  var alu33 = (acc0[4]+val0);
  var alu34 = (acc0[5]+val1);
  var alu35 = (acc0[6]+val2);
  var alu36 = (acc0[7]+val3);
  var alu37 = (acc0[8]+val0);
  var alu38 = (acc0[9]+val1);
  var alu39 = (acc0[10]+val2);
  var alu40 = (acc0[11]+val3);
  var alu41 = ((gidx0*192)+(lidx1*24)+alu12+(gidx1*4608)+(lidx0*2304));
  data0_18432[alu41] = (alu29*(1/(1.0f+exp2((alu29*-1.4426950408889634f)))));
  data0_18432[(alu41+576)] = (alu30*(1/(1.0f+exp2((alu30*-1.4426950408889634f)))));
  data0_18432[(alu41+1152)] = (alu31*(1/(1.0f+exp2((alu31*-1.4426950408889634f)))));
  data0_18432[(alu41+1728)] = (alu32*(1/(1.0f+exp2((alu32*-1.4426950408889634f)))));
  data0_18432[(alu41+1)] = (alu33*(1/(1.0f+exp2((alu33*-1.4426950408889634f)))));
  data0_18432[(alu41+577)] = (alu34*(1/(1.0f+exp2((alu34*-1.4426950408889634f)))));
  data0_18432[(alu41+1153)] = (alu35*(1/(1.0f+exp2((alu35*-1.4426950408889634f)))));
  data0_18432[(alu41+1729)] = (alu36*(1/(1.0f+exp2((alu36*-1.4426950408889634f)))));
  data0_18432[(alu41+2)] = (alu37*(1/(1.0f+exp2((alu37*-1.4426950408889634f)))));
  data0_18432[(alu41+578)] = (alu38*(1/(1.0f+exp2((alu38*-1.4426950408889634f)))));
  data0_18432[(alu41+1154)] = (alu39*(1/(1.0f+exp2((alu39*-1.4426950408889634f)))));
  data0_18432[(alu41+1730)] = (alu40*(1/(1.0f+exp2((alu40*-1.4426950408889634f)))));
}`;

const r_5_5_8_2_2_4_32_5_5_2_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_3200:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_18432:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_25600:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_32:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_32:array<f32>;
@group(0) @binding(7)var<storage,read_write>data6_32:array<f32>;
@group(0) @binding(8)var<storage,read_write>data7_32:array<f32>;
@compute @workgroup_size(8,2,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var lidx1 = i32(lindex.y); /* 2 */
  var lidx2 = i32(lindex.z); /* 2 */
  var gidx0 = i32(gindex.x); /* 5 */
  var gidx1 = i32(gindex.y); /* 5 */
  var lidx0 = i32(lindex.x); /* 8 */
  var precast0 = gidx0;
  var precast1 = lidx0;
  var precast2 = lidx2;
  var cast0 = bitcast<u32>(precast0);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  var precast3 = (cast0<<2u);
  var precast4 = (bitcast<u32>(precast1)<<2u);
  var cast1 = bitcast<i32>(precast4);
  var val0 = data3_32[cast1];
  var val1 = data4_32[cast1];
  var val2 = data5_32[cast1];
  var val3 = data6_32[cast1];
  var val4 = data7_32[cast1];
  var alu16 = (cast1+1);
  var val5 = data3_32[alu16];
  var val6 = data4_32[alu16];
  var val7 = data5_32[alu16];
  var val8 = data6_32[alu16];
  var val9 = data7_32[alu16];
  var alu17 = (cast1+2);
  var val10 = data3_32[alu17];
  var val11 = data7_32[alu17];
  var val12 = data4_32[alu17];
  var val13 = data5_32[alu17];
  var val14 = data6_32[alu17];
  var alu18 = (cast1+3);
  var val15 = data3_32[alu18];
  var val16 = data4_32[alu18];
  var val17 = data5_32[alu18];
  var val18 = data6_32[alu18];
  var val19 = data7_32[alu18];
  var precast5 = (bitcast<u32>(precast2)<<1u);
  for (var Ridx0 = 0; Ridx0 < 32; Ridx0++) {
    for (var Ridx1 = 0; Ridx1 < 5; Ridx1++) {
      for (var Ridx2 = 0; Ridx2 < 5; Ridx2++) {
        var alu19 = (bitcast<i32>(precast3)+bitcast<i32>(precast5)+Ridx2+(gidx1*96)+(lidx1*48)+(Ridx1*24)+(Ridx0*576));
        var val20 = data1_18432[alu19];
        var val21 = data1_18432[(alu19+1)];
        var val22 = data1_18432[(alu19+24)];
        var val23 = data1_18432[(alu19+25)];
        var alu20 = ((Ridx1*5)+Ridx2+(Ridx0*25)+(lidx0*3200));
        var val24 = data2_25600[alu20];
        var val25 = data2_25600[(alu20+800)];
        var val26 = data2_25600[(alu20+1600)];
        var val27 = data2_25600[(alu20+2400)];
        acc0[4] = (acc0[4]+(val20*val25));
        acc0[8] = (acc0[8]+(val20*val26));
        acc0[12] = (acc0[12]+(val20*val27));
        acc0[0] = (acc0[0]+(val20*val24));
        acc0[6] = (acc0[6]+(val21*val25));
        acc0[10] = (acc0[10]+(val21*val26));
        acc0[14] = (acc0[14]+(val21*val27));
        acc0[2] = (acc0[2]+(val21*val24));
        acc0[5] = (acc0[5]+(val22*val25));
        acc0[9] = (acc0[9]+(val22*val26));
        acc0[13] = (acc0[13]+(val22*val27));
        acc0[1] = (acc0[1]+(val22*val24));
        acc0[7] = (acc0[7]+(val23*val25));
        acc0[11] = (acc0[11]+(val23*val26));
        acc0[15] = (acc0[15]+(val23*val27));
        acc0[3] = (acc0[3]+(val23*val24));
      }
    }
  }
  var alu40 = (acc0[0]+val0);
  var alu41 = (1/sqrt((val3+1e-05f)));
  var alu42 = (acc0[1]+val0);
  var alu43 = (acc0[2]+val0);
  var alu44 = (acc0[3]+val0);
  var alu45 = (acc0[4]+val5);
  var alu46 = (1/sqrt((val8+1e-05f)));
  var alu47 = (acc0[5]+val5);
  var alu48 = (acc0[6]+val5);
  var alu49 = (acc0[7]+val5);
  var alu50 = (acc0[8]+val10);
  var alu51 = (1/sqrt((val14+1e-05f)));
  var alu52 = (acc0[9]+val10);
  var alu53 = (acc0[10]+val10);
  var alu54 = (acc0[11]+val10);
  var alu55 = (acc0[12]+val15);
  var alu56 = (1/sqrt((val18+1e-05f)));
  var alu57 = (acc0[13]+val15);
  var alu58 = (acc0[14]+val15);
  var alu59 = (acc0[15]+val15);
  var precast6 = (cast0<<1u);
  var alu60 = (lidx2+bitcast<i32>(precast6)+(gidx1*20)+(lidx1*10)+(lidx0*400));
  var alu61 = ((((alu40*(1/(1.0f+exp2((alu40*-1.4426950408889634f)))))-val1)*val2*alu41)+val4);
  var alu62 = ((((alu42*(1/(1.0f+exp2((alu42*-1.4426950408889634f)))))-val1)*val2*alu41)+val4);
  var alu63 = ((((alu43*(1/(1.0f+exp2((alu43*-1.4426950408889634f)))))-val1)*val2*alu41)+val4);
  var alu64 = ((((alu44*(1/(1.0f+exp2((alu44*-1.4426950408889634f)))))-val1)*val2*alu41)+val4);
  var alu65 = ((((alu45*(1/(1.0f+exp2((alu45*-1.4426950408889634f)))))-val6)*val7*alu46)+val9);
  var alu66 = ((((alu47*(1/(1.0f+exp2((alu47*-1.4426950408889634f)))))-val6)*val7*alu46)+val9);
  var alu67 = ((((alu48*(1/(1.0f+exp2((alu48*-1.4426950408889634f)))))-val6)*val7*alu46)+val9);
  var alu68 = ((((alu49*(1/(1.0f+exp2((alu49*-1.4426950408889634f)))))-val6)*val7*alu46)+val9);
  var alu69 = ((((alu50*(1/(1.0f+exp2((alu50*-1.4426950408889634f)))))-val12)*val13*alu51)+val11);
  var alu70 = ((((alu52*(1/(1.0f+exp2((alu52*-1.4426950408889634f)))))-val12)*val13*alu51)+val11);
  var alu71 = ((((alu53*(1/(1.0f+exp2((alu53*-1.4426950408889634f)))))-val12)*val13*alu51)+val11);
  var alu72 = ((((alu54*(1/(1.0f+exp2((alu54*-1.4426950408889634f)))))-val12)*val13*alu51)+val11);
  var alu73 = ((((alu55*(1/(1.0f+exp2((alu55*-1.4426950408889634f)))))-val16)*val17*alu56)+val19);
  var alu74 = ((((alu57*(1/(1.0f+exp2((alu57*-1.4426950408889634f)))))-val16)*val17*alu56)+val19);
  var alu75 = ((((alu58*(1/(1.0f+exp2((alu58*-1.4426950408889634f)))))-val16)*val17*alu56)+val19);
  var alu76 = ((((alu59*(1/(1.0f+exp2((alu59*-1.4426950408889634f)))))-val16)*val17*alu56)+val19);
  var alu77 = select(alu61,alu63,(alu61<alu63));
  var alu78 = select(alu65,alu67,(alu65<alu67));
  var alu79 = select(alu69,alu71,(alu69<alu71));
  var alu80 = select(alu73,alu75,(alu73<alu75));
  var alu81 = select(alu77,alu62,(alu77<alu62));
  var alu82 = select(alu78,alu66,(alu78<alu66));
  var alu83 = select(alu79,alu70,(alu79<alu70));
  var alu84 = select(alu80,alu74,(alu80<alu74));
  var alu85 = select(alu81,alu64,(alu81<alu64));
  data0_3200[alu60] = alu85;
  var alu87 = select(alu82,alu68,(alu82<alu68));
  data0_3200[(alu60+100)] = alu87;
  var alu89 = select(alu83,alu72,(alu83<alu72));
  data0_3200[(alu60+200)] = alu89;
  var alu91 = select(alu84,alu76,(alu84<alu76));
  data0_3200[(alu60+300)] = alu91;
}`;

const r_2_8_8_2_4_4_32_3_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_4096:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_3200:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_18432:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_64:array<f32>;
@compute @workgroup_size(8,8,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 2 */
  var lidx2 = i32(lindex.z); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 8 */
  var precast0 = gidx0;
  var precast1 = lidx0;
  var precast2 = lidx2;
  var cast0 = bitcast<u32>(precast0);
  var cast1 = bitcast<u32>(precast1);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  var precast3 = (cast0<<5u);
  var precast4 = (cast1<<2u);
  var alu16 = (bitcast<i32>(precast3)+bitcast<i32>(precast4));
  var val0 = data3_64[alu16];
  var val1 = data3_64[(alu16+1)];
  var val2 = data3_64[(alu16+2)];
  var val3 = data3_64[(alu16+3)];
  var precast5 = (bitcast<u32>(precast2)<<2u);
  var cast2 = bitcast<i32>(precast5);
  for (var Ridx0 = 0; Ridx0 < 32; Ridx0++) {
    var alu17 = ((gidx0*9216)+(lidx0*1152)+(Ridx0*9));
    var val4 = data2_18432[alu17];
    var val5 = data2_18432[(alu17+1)];
    var val6 = data2_18432[(alu17+2)];
    var val7 = data2_18432[(alu17+3)];
    var val8 = data2_18432[(alu17+4)];
    var val9 = data2_18432[(alu17+5)];
    var val10 = data2_18432[(alu17+6)];
    var val11 = data2_18432[(alu17+7)];
    var val12 = data2_18432[(alu17+8)];
    var val13 = data2_18432[(alu17+288)];
    var val14 = data2_18432[(alu17+289)];
    var val15 = data2_18432[(alu17+290)];
    var val16 = data2_18432[(alu17+291)];
    var val17 = data2_18432[(alu17+292)];
    var val18 = data2_18432[(alu17+293)];
    var val19 = data2_18432[(alu17+294)];
    var val20 = data2_18432[(alu17+295)];
    var val21 = data2_18432[(alu17+296)];
    var val22 = data2_18432[(alu17+576)];
    var val23 = data2_18432[(alu17+577)];
    var val24 = data2_18432[(alu17+578)];
    var val25 = data2_18432[(alu17+579)];
    var val26 = data2_18432[(alu17+580)];
    var val27 = data2_18432[(alu17+581)];
    var val28 = data2_18432[(alu17+582)];
    var val29 = data2_18432[(alu17+583)];
    var val30 = data2_18432[(alu17+584)];
    var val31 = data2_18432[(alu17+864)];
    var val32 = data2_18432[(alu17+865)];
    var val33 = data2_18432[(alu17+866)];
    var val34 = data2_18432[(alu17+867)];
    var val35 = data2_18432[(alu17+868)];
    var val36 = data2_18432[(alu17+869)];
    var val37 = data2_18432[(alu17+870)];
    var val38 = data2_18432[(alu17+871)];
    var val39 = data2_18432[(alu17+872)];
    var alu18 = ((lidx1*10)+cast2+(Ridx0*100));
    var val40 = data1_3200[alu18];
    var val41 = data1_3200[(alu18+1)];
    var val42 = data1_3200[(alu18+2)];
    var val43 = data1_3200[(alu18+3)];
    var val44 = data1_3200[(alu18+4)];
    var val45 = data1_3200[(alu18+5)];
    var val46 = data1_3200[(alu18+10)];
    var val47 = data1_3200[(alu18+11)];
    var val48 = data1_3200[(alu18+12)];
    var val49 = data1_3200[(alu18+13)];
    var val50 = data1_3200[(alu18+14)];
    var val51 = data1_3200[(alu18+15)];
    var val52 = data1_3200[(alu18+20)];
    var val53 = data1_3200[(alu18+21)];
    var val54 = data1_3200[(alu18+22)];
    var val55 = data1_3200[(alu18+23)];
    var val56 = data1_3200[(alu18+24)];
    var val57 = data1_3200[(alu18+25)];
    acc0[4] = (acc0[4]+(val41*val4)+(val42*val5)+(val43*val6)+(val47*val7)+(val48*val8)+(val49*val9)+(val53*val10)+(val54*val11)+(val55*val12));
    acc0[5] = (acc0[5]+(val41*val13)+(val42*val14)+(val43*val15)+(val47*val16)+(val48*val17)+(val49*val18)+(val53*val19)+(val54*val20)+(val55*val21));
    acc0[6] = (acc0[6]+(val41*val22)+(val42*val23)+(val43*val24)+(val47*val25)+(val48*val26)+(val49*val27)+(val53*val28)+(val54*val29)+(val55*val30));
    acc0[7] = (acc0[7]+(val41*val31)+(val42*val32)+(val43*val33)+(val47*val34)+(val48*val35)+(val49*val36)+(val53*val37)+(val54*val38)+(val55*val39));
    acc0[8] = (acc0[8]+(val42*val4)+(val43*val5)+(val44*val6)+(val48*val7)+(val49*val8)+(val50*val9)+(val54*val10)+(val55*val11)+(val56*val12));
    acc0[9] = (acc0[9]+(val42*val13)+(val43*val14)+(val44*val15)+(val48*val16)+(val49*val17)+(val50*val18)+(val54*val19)+(val55*val20)+(val56*val21));
    acc0[10] = (acc0[10]+(val42*val22)+(val43*val23)+(val44*val24)+(val48*val25)+(val49*val26)+(val50*val27)+(val54*val28)+(val55*val29)+(val56*val30));
    acc0[11] = (acc0[11]+(val42*val31)+(val43*val32)+(val44*val33)+(val48*val34)+(val49*val35)+(val50*val36)+(val54*val37)+(val55*val38)+(val56*val39));
    acc0[12] = (acc0[12]+(val43*val4)+(val44*val5)+(val45*val6)+(val49*val7)+(val50*val8)+(val51*val9)+(val55*val10)+(val56*val11)+(val57*val12));
    acc0[13] = (acc0[13]+(val43*val13)+(val44*val14)+(val45*val15)+(val49*val16)+(val50*val17)+(val51*val18)+(val55*val19)+(val56*val20)+(val57*val21));
    acc0[14] = (acc0[14]+(val43*val22)+(val44*val23)+(val45*val24)+(val49*val25)+(val50*val26)+(val51*val27)+(val55*val28)+(val56*val29)+(val57*val30));
    acc0[15] = (acc0[15]+(val43*val31)+(val44*val32)+(val45*val33)+(val49*val34)+(val50*val35)+(val51*val36)+(val55*val37)+(val56*val38)+(val57*val39));
    acc0[1] = (acc0[1]+(val40*val13)+(val41*val14)+(val42*val15)+(val46*val16)+(val47*val17)+(val48*val18)+(val52*val19)+(val53*val20)+(val54*val21));
    acc0[2] = (acc0[2]+(val40*val22)+(val41*val23)+(val42*val24)+(val46*val25)+(val47*val26)+(val48*val27)+(val52*val28)+(val53*val29)+(val54*val30));
    acc0[3] = (acc0[3]+(val40*val31)+(val41*val32)+(val42*val33)+(val46*val34)+(val47*val35)+(val48*val36)+(val52*val37)+(val53*val38)+(val54*val39));
    acc0[0] = (acc0[0]+(val40*val4)+(val41*val5)+(val42*val6)+(val46*val7)+(val47*val8)+(val48*val9)+(val52*val10)+(val53*val11)+(val54*val12));
  }
  var precast6 = lidx1;
  var alu36 = (acc0[0]+val0);
  var alu37 = (acc0[1]+val1);
  var alu38 = (acc0[2]+val2);
  var alu39 = (acc0[3]+val3);
  var alu40 = (acc0[4]+val0);
  var alu41 = (acc0[5]+val1);
  var alu42 = (acc0[6]+val2);
  var alu43 = (acc0[7]+val3);
  var alu44 = (acc0[8]+val0);
  var alu45 = (acc0[9]+val1);
  var alu46 = (acc0[10]+val2);
  var alu47 = (acc0[11]+val3);
  var alu48 = (acc0[12]+val0);
  var alu49 = (acc0[13]+val1);
  var alu50 = (acc0[14]+val2);
  var alu51 = (acc0[15]+val3);
  var precast7 = (cast0<<11u);
  var precast8 = (cast1<<8u);
  var precast9 = (bitcast<u32>(precast6)<<3u);
  var alu52 = (bitcast<i32>(precast7)+bitcast<i32>(precast8)+bitcast<i32>(precast9)+cast2);
  data0_4096[alu52] = (alu36*(1/(1.0f+exp2((alu36*-1.4426950408889634f)))));
  data0_4096[(alu52+1)] = (alu40*(1/(1.0f+exp2((alu40*-1.4426950408889634f)))));
  data0_4096[(alu52+2)] = (alu44*(1/(1.0f+exp2((alu44*-1.4426950408889634f)))));
  data0_4096[(alu52+3)] = (alu48*(1/(1.0f+exp2((alu48*-1.4426950408889634f)))));
  data0_4096[(alu52+64)] = (alu37*(1/(1.0f+exp2((alu37*-1.4426950408889634f)))));
  data0_4096[(alu52+65)] = (alu41*(1/(1.0f+exp2((alu41*-1.4426950408889634f)))));
  data0_4096[(alu52+66)] = (alu45*(1/(1.0f+exp2((alu45*-1.4426950408889634f)))));
  data0_4096[(alu52+67)] = (alu49*(1/(1.0f+exp2((alu49*-1.4426950408889634f)))));
  data0_4096[(alu52+128)] = (alu38*(1/(1.0f+exp2((alu38*-1.4426950408889634f)))));
  data0_4096[(alu52+129)] = (alu42*(1/(1.0f+exp2((alu42*-1.4426950408889634f)))));
  data0_4096[(alu52+130)] = (alu46*(1/(1.0f+exp2((alu46*-1.4426950408889634f)))));
  data0_4096[(alu52+131)] = (alu50*(1/(1.0f+exp2((alu50*-1.4426950408889634f)))));
  data0_4096[(alu52+192)] = (alu39*(1/(1.0f+exp2((alu39*-1.4426950408889634f)))));
  data0_4096[(alu52+193)] = (alu43*(1/(1.0f+exp2((alu43*-1.4426950408889634f)))));
  data0_4096[(alu52+194)] = (alu47*(1/(1.0f+exp2((alu47*-1.4426950408889634f)))));
  data0_4096[(alu52+195)] = (alu51*(1/(1.0f+exp2((alu51*-1.4426950408889634f)))));
}`;

const r_8_8_3_3_64_3_3_2_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_576:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_4096:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_36864:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_64:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_64:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_64:array<f32>;
@group(0) @binding(7)var<storage,read_write>data6_64:array<f32>;
@group(0) @binding(8)var<storage,read_write>data7_64:array<f32>;
@compute @workgroup_size(8,3,3) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,4>;
  var lidx1 = i32(lindex.y); /* 3 */
  var lidx2 = i32(lindex.z); /* 3 */
  var gidx0 = i32(gindex.x); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var precast0 = gidx0;
  var precast1 = lidx1;
  var precast2 = lidx2;
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  var precast3 = (bitcast<u32>(precast0)<<3u);
  var alu4 = (lidx0+bitcast<i32>(precast3));
  var val0 = data3_64[alu4];
  var val1 = data4_64[alu4];
  var val2 = data5_64[alu4];
  var val3 = data6_64[alu4];
  var val4 = data7_64[alu4];
  var precast4 = (bitcast<u32>(precast1)<<4u);
  var precast5 = (bitcast<u32>(precast2)<<1u);
  for (var Ridx0 = 0; Ridx0 < 64; Ridx0++) {
    var precast6 = Ridx0;
    var precast7 = (bitcast<u32>(precast6)<<6u);
    for (var Ridx1 = 0; Ridx1 < 3; Ridx1++) {
      var precast8 = Ridx1;
      var precast9 = (bitcast<u32>(precast8)<<3u);
      for (var Ridx2 = 0; Ridx2 < 3; Ridx2++) {
        var alu5 = (bitcast<i32>(precast4)+bitcast<i32>(precast9)+bitcast<i32>(precast5)+Ridx2+bitcast<i32>(precast7));
        var val5 = data1_4096[alu5];
        var val6 = data1_4096[(alu5+1)];
        var val7 = data1_4096[(alu5+8)];
        var val8 = data1_4096[(alu5+9)];
        var val9 = data2_36864[((Ridx1*3)+Ridx2+(Ridx0*9)+(gidx0*4608)+(lidx0*576))];
        acc0[0] = (acc0[0]+(val5*val9));
        acc0[2] = (acc0[2]+(val6*val9));
        acc0[1] = (acc0[1]+(val7*val9));
        acc0[3] = (acc0[3]+(val8*val9));
      }
    }
  }
  var alu13 = (acc0[0]+val0);
  var alu14 = (1/sqrt((val3+1e-05f)));
  var alu15 = (acc0[1]+val0);
  var alu16 = (acc0[2]+val0);
  var alu17 = (acc0[3]+val0);
  var alu18 = ((((alu13*(1/(1.0f+exp2((alu13*-1.4426950408889634f)))))-val1)*val2*alu14)+val4);
  var alu19 = ((((alu15*(1/(1.0f+exp2((alu15*-1.4426950408889634f)))))-val1)*val2*alu14)+val4);
  var alu20 = ((((alu16*(1/(1.0f+exp2((alu16*-1.4426950408889634f)))))-val1)*val2*alu14)+val4);
  var alu21 = ((((alu17*(1/(1.0f+exp2((alu17*-1.4426950408889634f)))))-val1)*val2*alu14)+val4);
  var alu22 = select(alu18,alu20,(alu18<alu20));
  var alu23 = select(alu22,alu19,(alu22<alu19));
  var alu24 = select(alu23,alu21,(alu23<alu21));
  data0_576[(lidx2+(lidx1*3)+(gidx0*72)+(lidx0*9))] = alu24;
}`;

const r_10_16_36 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_10:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_576:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_5760:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_10:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 10 */
  var lidx0 = i32(lindex.x); /* 16 */
  acc0[0] = 0.0f;
  acc1[0] = 0.0f;
  var val0 = data3_10[gidx0];
  for (var Ridx0_0 = 0; Ridx0_0 < 36; Ridx0_0++) {
    var alu2 = ((lidx0*36)+Ridx0_0);
    var val1 = data1_576[alu2];
    var val2 = data2_5760[(alu2+(gidx0*576))];
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

    const layouts=[device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]})]

    const buf_0 = createEmptyBuf(device, 73728);;
    const input0 = createEmptyBuf(device, 3136);;
    const buf_1 = createWeightBuf(device, 3200, getTensorBuffer(safetensor, metadata['layers.0.weight']));
    const buf_2 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['layers.0.bias']));
    const buf_3 = createEmptyBuf(device, 12800);;
    const buf_4 = createWeightBuf(device, 102400, getTensorBuffer(safetensor, metadata['layers.2.weight']));
    const buf_5 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['layers.2.bias']));
    const buf_6 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['layers.4.running_mean']));
    const buf_7 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['layers.4.weight']));
    const buf_8 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['layers.4.running_var']));
    const buf_9 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['layers.4.bias']));
    const buf_10 = createEmptyBuf(device, 16384);;
    const buf_11 = createWeightBuf(device, 73728, getTensorBuffer(safetensor, metadata['layers.6.weight']));
    const buf_12 = createWeightBuf(device, 256, getTensorBuffer(safetensor, metadata['layers.6.bias']));
    const buf_13 = createEmptyBuf(device, 2304);;
    const buf_14 = createWeightBuf(device, 147456, getTensorBuffer(safetensor, metadata['layers.8.weight']));
    const buf_15 = createWeightBuf(device, 256, getTensorBuffer(safetensor, metadata['layers.8.bias']));
    const buf_16 = createWeightBuf(device, 256, getTensorBuffer(safetensor, metadata['layers.10.running_mean']));
    const buf_17 = createWeightBuf(device, 256, getTensorBuffer(safetensor, metadata['layers.10.weight']));
    const buf_18 = createWeightBuf(device, 256, getTensorBuffer(safetensor, metadata['layers.10.running_var']));
    const buf_19 = createWeightBuf(device, 256, getTensorBuffer(safetensor, metadata['layers.10.bias']));
    const output0 = createEmptyBuf(device, 40);;
    const buf_20 = createWeightBuf(device, 23040, getTensorBuffer(safetensor, metadata['layers.13.weight']));
    const buf_21 = createWeightBuf(device, 40, getTensorBuffer(safetensor, metadata['layers.13.bias']));

    const gpuWriteBuffer0 = device.createBuffer({size:input0.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });

    const gpuReadBuffer0 = device.createBuffer({size:output0.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const kernels = [r_4_3_2_8_8_3_4_5_5, r_5_5_8_2_2_4_32_5_5_2_2, r_2_8_8_2_4_4_32_3_3, r_8_8_3_3_64_3_3_2_2, r_10_16_36];
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
        addComputePass(device, commandEncoder, pipelines[0], layouts[0], infinityBuf, [buf_0, input0, buf_1, buf_2], [3, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[1], layouts[1], infinityBuf, [buf_3, buf_0, buf_4, buf_5, buf_6, buf_7, buf_8, buf_9], [5, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[2], layouts[2], infinityBuf, [buf_10, buf_3, buf_11, buf_12], [2, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[3], layouts[3], infinityBuf, [buf_13, buf_10, buf_14, buf_15, buf_16, buf_17, buf_18, buf_19], [8, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[4], layouts[4], infinityBuf, [output0, buf_13, buf_20, buf_21], [10, 1, 1]);
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
export default mnist_convnet;
