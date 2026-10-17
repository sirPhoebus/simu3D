struct Camera {
    view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    sun_dir: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: Camera;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
}

@vertex
fn vs_sky(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(pos[index], 0.9999, 1.0);
}

@fragment
fn fs_sky(@builtin(position) clip: vec4<f32>) -> @location(0) vec4<f32> {
    let t = clamp(clip.y / 900.0, 0.0, 1.0);
    let horizon = vec3<f32>(0.78, 0.72, 0.62);
    let zenith = vec3<f32>(0.42, 0.62, 0.84);
    return vec4<f32>(mix(horizon, zenith, t), 1.0);
}

@vertex
fn vs_static(
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
) -> VsOut {
    var out: VsOut;
    out.world = pos;
    out.normal = normal;
    out.color = color;
    out.clip = camera.view_proj * vec4<f32>(pos, 1.0);
    return out;
}

@vertex
fn vs_instance(
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) c0: vec4<f32>,
    @location(3) c1: vec4<f32>,
    @location(4) c2: vec4<f32>,
    @location(5) c3: vec4<f32>,
    @location(6) color: vec4<f32>,
) -> VsOut {
    let model = mat4x4<f32>(c0, c1, c2, c3);
    let world = (model * vec4<f32>(pos, 1.0)).xyz;
    let n = normalize((model * vec4<f32>(normal, 0.0)).xyz);
    var out: VsOut;
    out.world = world;
    out.normal = n;
    out.color = color;
    out.clip = camera.view_proj * vec4<f32>(world, 1.0);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let sun = normalize(camera.sun_dir.xyz);
    let ndl = max(dot(n, sun), 0.0);
    let hemi = n.y * 0.5 + 0.5;
    let sky = mix(vec3<f32>(0.42, 0.38, 0.32), vec3<f32>(0.62, 0.74, 0.90), hemi);
    let sun_col = vec3<f32>(1.0, 0.94, 0.82);
    var lit = in.color.rgb * (sky * 0.55 + sun_col * ndl * 0.7 + vec3<f32>(0.08));
    let edge = length(fwidth(n));
    lit = lit * mix(1.0, 0.45, smoothstep(0.01, 0.15, edge));
    lit = lit + in.color.rgb * vec3<f32>(0.16, 0.13, 0.09) * (1.0 - hemi) * 0.35;
    let view = normalize(camera.cam_pos.xyz - in.world);
    if (in.color.a < 0.95) {
        let fres = pow(1.0 - max(dot(n, view), 0.0), 3.0);
        lit = mix(lit, vec3<f32>(0.86, 0.92, 0.95), fres * 0.45);
        let spec = pow(max(dot(reflect(-sun, n), view), 0.0), 40.0);
        lit = lit + sun_col * spec * 0.28;
    }
    return vec4<f32>(lit, in.color.a);
}

@vertex
fn vs_ui(@location(0) pos: vec2<f32>, @location(1) color: vec4<f32>) -> VsOut {
    var out: VsOut;
    out.clip = vec4<f32>(pos, 0.0, 1.0);
    out.world = vec3<f32>(0.0);
    out.normal = vec3<f32>(0.0, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_ui(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
