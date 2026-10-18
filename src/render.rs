//! wgpu renderer. The standing house is a greedy mesh. Debris, falling structure, and the rocket are instanced boxes.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::event_loop::OwnedDisplayHandle;
use winit::window::Window;

use simu3d::math3::Vec3;
use simu3d::mesh::{Mesh, Vertex as CpuVertex};
use simu3d::sim::Instance;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    sun_dir: [f32; 4],
    sky_right: [f32; 4],
    sky_up: [f32; 4],
    sky_forward: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceRaw {
    c0: [f32; 4],
    c1: [f32; 4],
    c2: [f32; 4],
    c3: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UiVertex {
    pos: [f32; 2],
    color: [f32; 4],
}

struct GpuMesh {
    vertex: wgpu::Buffer,
    index: wgpu::Buffer,
    count: u32,
}

pub struct Gpu {
    pub adapter_name: String,
    pub backend: wgpu::Backend,
    window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
    view_format: wgpu::TextureFormat,
    size: PhysicalSize<u32>,
    depth: wgpu::TextureView,
    camera_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    opaque: wgpu::RenderPipeline,
    transparent: wgpu::RenderPipeline,
    instance_opaque: wgpu::RenderPipeline,
    instance_transparent: wgpu::RenderPipeline,
    sky: wgpu::RenderPipeline,
    ui: wgpu::RenderPipeline,
    static_opaque: Option<GpuMesh>,
    static_transparent: Option<GpuMesh>,
    cube_vertex: wgpu::Buffer,
    cube_index: wgpu::Buffer,
    cube_index_count: u32,
    cube_bevel: f32,
    instance_buf: wgpu::Buffer,
    instance_capacity: usize,
    ui_buf: wgpu::Buffer,
}

impl Gpu {
    pub async fn new(display: OwnedDisplayHandle, window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle(
            Box::new(display),
        ));
        let size = window.inner_size();
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
        if adapters.is_empty() {
            panic!("no graphics adapter");
        }
        println!("Adapters:");
        for candidate in &adapters {
            let info = candidate.get_info();
            println!(
                "  {} | {:?} | {:?} | surface={}",
                info.name,
                info.device_type,
                info.backend,
                candidate.is_surface_supported(&surface)
            );
        }
        let adapter = pick_adapter(adapters, &surface);
        let info = adapter.get_info();
        if matches!(info.device_type, wgpu::DeviceType::Cpu) {
            println!(
                "SOFTWARE RENDERER: {}. This process is not using the RTX 4090.",
                info.name
            );
        } else {
            println!("GPU in use: {} ({:?})", info.name, info.backend);
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("device");
        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps.formats[0];
        let view_format = surface_format.add_srgb_suffix();
        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sky_pixels = simu3d::sky::equirect_rgba16(512, 256);
        let sky_tex = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("sky"),
                size: wgpu::Extent3d {
                    width: 512,
                    height: 256,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&sky_pixels),
        );
        let sky_view = sky_tex.create_view(&Default::default());
        let sky_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sky-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera-group"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&sky_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sky_sampler),
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let opaque = pipeline(
            &device,
            &layout,
            &shader,
            "vs_static",
            "fs_main",
            view_format,
            &[Some(GpuVertex::layout())],
            true,
            Some(wgpu::Face::Back),
            false,
        );
        let transparent = pipeline(
            &device,
            &layout,
            &shader,
            "vs_static",
            "fs_main",
            view_format,
            &[Some(GpuVertex::layout())],
            false,
            None,
            true,
        );
        let instance_opaque = pipeline(
            &device,
            &layout,
            &shader,
            "vs_instance",
            "fs_main",
            view_format,
            &[Some(unit_cube_layout()), Some(InstanceRaw::layout())],
            true,
            Some(wgpu::Face::Back),
            false,
        );
        let instance_transparent = pipeline(
            &device,
            &layout,
            &shader,
            "vs_instance",
            "fs_main",
            view_format,
            &[Some(unit_cube_layout()), Some(InstanceRaw::layout())],
            false,
            None,
            true,
        );
        let sky = pipeline(
            &device,
            &layout,
            &shader,
            "vs_sky",
            "fs_sky",
            view_format,
            &[],
            false,
            None,
            false,
        );
        let ui = ui_pipeline(&device, &layout, &shader, view_format);
        let (cube_vertex, cube_index, cube_index_count) = unit_cube_buffers(&device, 0.006);
        let instance_capacity = 2048;
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instances"),
            size: (instance_capacity * std::mem::size_of::<InstanceRaw>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ui_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui"),
            size: (65536 * std::mem::size_of::<UiVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let depth = depth_view(&device, size.width.max(1), size.height.max(1));
        let mut gpu = Self {
            adapter_name: info.name,
            backend: info.backend,
            window,
            device,
            queue,
            surface,
            surface_format,
            view_format,
            size,
            depth,
            camera_buf,
            bind_group,
            opaque,
            transparent,
            instance_opaque,
            instance_transparent,
            sky,
            ui,
            static_opaque: None,
            static_transparent: None,
            cube_vertex,
            cube_index,
            cube_index_count,
            cube_bevel: 0.006,
            instance_buf,
            instance_capacity,
            ui_buf,
        };
        gpu.configure();
        gpu
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.size = size;
        self.configure();
    }

    pub fn aspect(&self) -> f32 {
        let h = self.size.height.max(1) as f32;
        self.size.width as f32 / h
    }

    pub fn upload_static(&mut self, opaque: &Mesh, transparent: &Mesh) {
        self.static_opaque = upload_mesh(&self.device, opaque, "static-opaque");
        self.static_transparent = upload_mesh(&self.device, transparent, "static-transparent");
    }

    pub fn set_block_gap(&mut self, gap: f32) {
        let bevel = (gap * 1.5).clamp(0.0, 0.4);
        if (bevel - self.cube_bevel).abs() < 1.0e-5 {
            return;
        }
        self.cube_bevel = bevel;
        let (vertex, index, count) = unit_cube_buffers(&self.device, bevel);
        self.cube_vertex = vertex;
        self.cube_index = index;
        self.cube_index_count = count;
    }

    pub fn render(
        &mut self,
        eye: Vec3,
        look: Vec3,
        aspect: f32,
        instances: &[Instance],
        chamber: u32,
        reserve: u32,
        reloading: bool,
        show_reticle: bool,
        fov_degrees: f32,
        settings: &[SettingLine],
    ) {
        if self.size.width == 0 || self.size.height == 0 {
            return;
        }
        let view = look_to_rh(eye, look, Vec3::Y);
        let fov_y = fov_degrees.to_radians().clamp(0.4, 2.2);
        let proj = perspective_rh(fov_y, aspect, 0.08, 1800.0);
        let view_proj = mul(proj, view);
        let forward = look.normalized();
        let right = forward.cross(Vec3::Y).normalized();
        let up = right.cross(forward);
        let sun = simu3d::sky::sun_direction();
        let uniform = CameraUniform {
            view_proj,
            cam_pos: [eye.x, eye.y, eye.z, 1.0],
            sun_dir: [sun.x, sun.y, sun.z, 0.0],
            sky_right: [right.x, right.y, right.z, self.size.width as f32],
            sky_up: [up.x, up.y, up.z, self.size.height as f32],
            sky_forward: [forward.x, forward.y, forward.z, (fov_y * 0.5).tan()],
        };
        self.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&uniform));

        let (opaque_n, transparent_n) = self.write_instances(instances);
        let ui_count = self.write_ui(chamber, reserve, reloading, show_reticle, settings);

        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
                drop(texture);
                self.configure();
                return;
            }
            wgpu::CurrentSurfaceTexture::Occluded | wgpu::CurrentSurfaceTexture::Timeout => return,
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.configure();
                return;
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                // Recreated on the next successful resize/resumed path. Skip this frame.
                return;
            }
            wgpu::CurrentSurfaceTexture::Validation => return,
        };
        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                format: Some(self.view_format),
                ..Default::default()
            });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frame"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.55,
                            g: 0.68,
                            b: 0.82,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_pipeline(&self.sky);
            pass.draw(0..3, 0..1);
            if let Some(mesh) = &self.static_opaque {
                pass.set_pipeline(&self.opaque);
                pass.set_vertex_buffer(0, mesh.vertex.slice(..));
                pass.set_index_buffer(mesh.index.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.count, 0, 0..1);
            }
            if opaque_n > 0 {
                pass.set_pipeline(&self.instance_opaque);
                pass.set_vertex_buffer(0, self.cube_vertex.slice(..));
                pass.set_vertex_buffer(1, self.instance_buf.slice(..));
                pass.set_index_buffer(self.cube_index.slice(..), wgpu::IndexFormat::Uint16);
                pass.draw_indexed(0..self.cube_index_count, 0, 0..opaque_n);
            }
            if let Some(mesh) = &self.static_transparent {
                pass.set_pipeline(&self.transparent);
                pass.set_vertex_buffer(0, mesh.vertex.slice(..));
                pass.set_index_buffer(mesh.index.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.count, 0, 0..1);
            }
            if transparent_n > 0 {
                pass.set_pipeline(&self.instance_transparent);
                pass.set_vertex_buffer(0, self.cube_vertex.slice(..));
                pass.set_vertex_buffer(1, self.instance_buf.slice(..));
                pass.set_index_buffer(self.cube_index.slice(..), wgpu::IndexFormat::Uint16);
                // Transparent instances are packed after the opaque ones.
                pass.draw_indexed(
                    0..self.cube_index_count,
                    0,
                    opaque_n..opaque_n + transparent_n,
                );
            }
            if ui_count > 0 {
                pass.set_pipeline(&self.ui);
                pass.set_vertex_buffer(0, self.ui_buf.slice(..));
                pass.draw(0..ui_count, 0..1);
            }
        }
        self.queue.submit([encoder.finish()]);
        self.window.pre_present_notify();
        self.queue.present(surface_texture);
    }

    fn write_instances(&mut self, instances: &[Instance]) -> (u32, u32) {
        let mut opaque = Vec::new();
        let mut transparent = Vec::new();
        for instance in instances {
            let raw = InstanceRaw::from_instance(instance);
            if instance.transparent {
                transparent.push(raw);
            } else {
                opaque.push(raw);
            }
        }
        let needed = opaque.len() + transparent.len();
        if needed > self.instance_capacity {
            self.instance_capacity = needed.next_power_of_two();
            self.instance_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instances"),
                size: (self.instance_capacity * std::mem::size_of::<InstanceRaw>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        let mut packed = opaque;
        let opaque_n = packed.len() as u32;
        packed.extend(transparent);
        let transparent_n = packed.len() as u32 - opaque_n;
        if !packed.is_empty() {
            self.queue
                .write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(&packed));
        }
        (opaque_n, transparent_n)
    }

    fn write_ui(
        &mut self,
        chamber: u32,
        reserve: u32,
        reloading: bool,
        show_reticle: bool,
        settings: &[SettingLine],
    ) -> u32 {
        let mut verts = Vec::new();
        let cross = if reloading {
            [0.85, 0.25, 0.18, 0.9]
        } else if chamber == 0 && reserve == 0 {
            [0.55, 0.2, 0.16, 0.9]
        } else {
            [0.95, 0.95, 0.92, 0.85]
        };
        if show_reticle {
            push_ndc_quad(&mut verts, -0.012, -0.0015, 0.012, 0.0015, cross);
            push_ndc_quad(&mut verts, -0.0015, -0.02, 0.0015, 0.02, cross);
        }
        let pip = [0.92, 0.78, 0.35, 0.95];
        let empty = [0.25, 0.25, 0.25, 0.7];
        if chamber > 0 {
            push_ndc_quad(&mut verts, -0.92, -0.90, -0.86, -0.86, pip);
        }
        for i in 0..reserve.min(12) {
            let x = -0.84 + i as f32 * 0.035;
            push_ndc_quad(&mut verts, x, -0.895, x + 0.022, -0.87, empty);
        }
        if chamber == 0 && reserve == 0 && !reloading {
            let width = self.size.width.max(1) as f32;
            let height = self.size.height.max(1) as f32;
            push_text(
                &mut verts,
                "EMPTY  R",
                -0.92,
                -0.82,
                2.0,
                [0.92, 0.35, 0.28, 1.0],
                width,
                height,
            );
        }
        if !settings.is_empty() {
            push_ndc_quad(
                &mut verts,
                -0.98,
                -0.95,
                -0.22,
                0.95,
                [0.05, 0.06, 0.07, 0.92],
            );
            let width = self.size.width.max(1) as f32;
            let height = self.size.height.max(1) as f32;
            push_text(
                &mut verts,
                "SETTINGS  ARROWS  WHEEL",
                -0.96,
                0.88,
                2.0,
                [0.95, 0.86, 0.55, 1.0],
                width,
                height,
            );
            push_text(
                &mut verts,
                "SELECT RELOAD THEN ENTER",
                -0.96,
                0.82,
                1.5,
                [0.70, 0.70, 0.68, 1.0],
                width,
                height,
            );
            for (index, line) in settings.iter().enumerate() {
                let y = 0.74 - index as f32 * 0.055;
                if line.selected {
                    push_ndc_quad(
                        &mut verts,
                        -0.96,
                        y - 0.04,
                        -0.24,
                        y + 0.012,
                        [0.25, 0.22, 0.12, 0.95],
                    );
                }
                push_text(
                    &mut verts,
                    &line.text,
                    -0.95,
                    y,
                    2.0,
                    if line.selected {
                        [1.0, 0.95, 0.75, 1.0]
                    } else {
                        [0.86, 0.88, 0.86, 1.0]
                    },
                    width,
                    height,
                );
            }
        }
        if verts.is_empty() {
            return 0;
        }
        let bytes = bytemuck::cast_slice(&verts);
        if bytes.len() as u64 > self.ui_buf.size() {
            self.ui_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ui"),
                size: bytes.len().next_power_of_two() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        self.queue.write_buffer(&self.ui_buf, 0, bytes);
        verts.len() as u32
    }

    fn configure(&mut self) {
        self.surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: self.surface_format,
                color_space: wgpu::SurfaceColorSpace::Auto,
                view_formats: vec![self.view_format],
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                width: self.size.width.max(1),
                height: self.size.height.max(1),
                desired_maximum_frame_latency: 2,
                present_mode: wgpu::PresentMode::AutoVsync,
            },
        );
        self.depth = depth_view(
            &self.device,
            self.size.width.max(1),
            self.size.height.max(1),
        );
    }
}

impl GpuVertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 24,
                    shader_location: 2,
                },
            ],
        }
    }
}

impl InstanceRaw {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceRaw>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 3,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 4,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 48,
                    shader_location: 5,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 64,
                    shader_location: 6,
                },
            ],
        }
    }

    fn from_instance(instance: &Instance) -> Self {
        let m = model_matrix(instance.center, instance.half, instance.rotation_xyzw);
        Self {
            c0: m[0],
            c1: m[1],
            c2: m[2],
            c3: m[3],
            color: instance.color,
        }
    }
}

fn unit_cube_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: 24,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
        ],
    }
}

fn pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    vs: &str,
    fs: &str,
    format: wgpu::TextureFormat,
    buffers: &[Option<wgpu::VertexBufferLayout>],
    depth_write: bool,
    cull: Option<wgpu::Face>,
    blend: bool,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(vs),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(vs),
            compilation_options: Default::default(),
            buffers,
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fs),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: if blend {
                    Some(wgpu::BlendState::ALPHA_BLENDING)
                } else {
                    Some(wgpu::BlendState::REPLACE)
                },
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            cull_mode: cull,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(depth_write),
            depth_compare: Some(if vs == "vs_sky" {
                wgpu::CompareFunction::Always
            } else {
                wgpu::CompareFunction::Less
            }),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn ui_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("ui"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_ui"),
            compilation_options: Default::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<UiVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 8,
                        shader_location: 1,
                    },
                ],
            })],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_ui"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn upload_mesh(device: &wgpu::Device, mesh: &Mesh, label: &str) -> Option<GpuMesh> {
    if mesh.indices.is_empty() {
        return None;
    }
    let vertices: Vec<GpuVertex> = mesh
        .vertices
        .iter()
        .map(|v: &CpuVertex| GpuVertex {
            position: v.position,
            normal: v.normal,
            color: v.color,
        })
        .collect();
    let vertex = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&mesh.indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    Some(GpuMesh {
        vertex,
        index,
        count: mesh.indices.len() as u32,
    })
}

fn unit_cube_buffers(device: &wgpu::Device, chamfer: f32) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    let mut verts = Vec::new();
    let mut indices = Vec::new();
    push_beveled_cube(&mut verts, &mut indices, chamfer);
    let vertex = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("unit-cube"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("unit-cube-index"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    (vertex, index, indices.len() as u32)
}

fn push_beveled_cube(verts: &mut Vec<f32>, indices: &mut Vec<u16>, chamfer: f32) {
    let inset = 0.5 - chamfer;
    for axis in 0..3 {
        for positive in [false, true] {
            let mut normal = [0.0; 3];
            normal[axis] = if positive { 1.0 } else { -1.0 };
            let plane = normal[axis] * 0.5;
            let u = (axis + 1) % 3;
            let v = (axis + 2) % 3;
            let mut corners = [[0.0; 3]; 4];
            for (i, (ou, ov)) in [
                (-inset, -inset),
                (inset, -inset),
                (inset, inset),
                (-inset, inset),
            ]
            .into_iter()
            .enumerate()
            {
                let mut p = [0.0; 3];
                p[axis] = plane;
                p[u] = ou;
                p[v] = ov;
                corners[i] = p;
            }
            orient_quad(&mut corners, normal);
            push_cube_quad(verts, indices, corners, normal);
        }
    }
    for edge_axis in 0..3 {
        let a = (edge_axis + 1) % 3;
        let b = (edge_axis + 2) % 3;
        for sa in [-1.0_f32, 1.0] {
            for sb in [-1.0_f32, 1.0] {
                let mut normal = [0.0; 3];
                normal[a] = sa;
                normal[b] = sb;
                let len = (2.0_f32).sqrt();
                normal[a] /= len;
                normal[b] /= len;
                let point = |along: f32, on_a: bool| {
                    let mut q = [0.0; 3];
                    q[edge_axis] = along;
                    if on_a {
                        q[a] = sa * 0.5;
                        q[b] = sb * inset;
                    } else {
                        q[a] = sa * inset;
                        q[b] = sb * 0.5;
                    }
                    q
                };
                let mut corners = [
                    point(-inset, true),
                    point(inset, true),
                    point(inset, false),
                    point(-inset, false),
                ];
                orient_quad(&mut corners, normal);
                push_cube_quad(verts, indices, corners, normal);
            }
        }
    }
    let s3 = (3.0_f32).sqrt();
    for sx in [-1.0_f32, 1.0] {
        for sy in [-1.0_f32, 1.0] {
            for sz in [-1.0_f32, 1.0] {
                let signs = [sx, sy, sz];
                let mut pts = [[0.0; 3]; 3];
                for axis in 0..3 {
                    let mut p = [0.0; 3];
                    for k in 0..3 {
                        p[k] = if k == axis {
                            signs[k] * 0.5
                        } else {
                            signs[k] * inset
                        };
                    }
                    pts[axis] = p;
                }
                let normal = [sx / s3, sy / s3, sz / s3];
                orient_tri(&mut pts, normal);
                push_cube_tri(verts, indices, pts, normal);
            }
        }
    }
}

fn orient_quad(corners: &mut [[f32; 3]; 4], normal: [f32; 3]) {
    let e1 = [
        corners[1][0] - corners[0][0],
        corners[1][1] - corners[0][1],
        corners[1][2] - corners[0][2],
    ];
    let e2 = [
        corners[2][0] - corners[0][0],
        corners[2][1] - corners[0][1],
        corners[2][2] - corners[0][2],
    ];
    let cross = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let dot = cross[0] * normal[0] + cross[1] * normal[1] + cross[2] * normal[2];
    if dot < 0.0 {
        corners.swap(1, 3);
    }
}

fn orient_tri(pts: &mut [[f32; 3]; 3], normal: [f32; 3]) {
    let e1 = [
        pts[1][0] - pts[0][0],
        pts[1][1] - pts[0][1],
        pts[1][2] - pts[0][2],
    ];
    let e2 = [
        pts[2][0] - pts[0][0],
        pts[2][1] - pts[0][1],
        pts[2][2] - pts[0][2],
    ];
    let cross = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let dot = cross[0] * normal[0] + cross[1] * normal[1] + cross[2] * normal[2];
    if dot < 0.0 {
        pts.swap(1, 2);
    }
}

fn push_cube_quad(
    verts: &mut Vec<f32>,
    indices: &mut Vec<u16>,
    corners: [[f32; 3]; 4],
    normal: [f32; 3],
) {
    let base = (verts.len() / 6) as u16;
    for corner in corners {
        verts.extend_from_slice(&corner);
        verts.extend_from_slice(&normal);
    }
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn push_cube_tri(
    verts: &mut Vec<f32>,
    indices: &mut Vec<u16>,
    pts: [[f32; 3]; 3],
    normal: [f32; 3],
) {
    let base = (verts.len() / 6) as u16;
    for corner in pts {
        verts.extend_from_slice(&corner);
        verts.extend_from_slice(&normal);
    }
    indices.extend_from_slice(&[base, base + 1, base + 2]);
}

fn depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&Default::default())
}

fn pick_adapter(mut adapters: Vec<wgpu::Adapter>, surface: &wgpu::Surface) -> wgpu::Adapter {
    adapters.retain(|adapter| adapter.is_surface_supported(surface));
    adapters.sort_by_key(|adapter| match adapter.get_info().device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Other => 3,
        wgpu::DeviceType::Cpu => 4,
    });
    adapters
        .into_iter()
        .next()
        .expect("no adapter can draw this window")
}

pub struct SettingLine {
    pub text: String,
    pub selected: bool,
}

fn push_text(
    verts: &mut Vec<UiVertex>,
    text: &str,
    mut x: f32,
    y: f32,
    scale: f32,
    color: [f32; 4],
    width: f32,
    height: f32,
) {
    let px = 2.0 / width * scale;
    let py = 2.0 / height * scale;
    for ch in text.chars() {
        let rows = crate::bitmap_font::rows(ch);
        for (row, bits) in rows.iter().enumerate() {
            for col in 0..5 {
                if bits & (1 << (4 - col)) != 0 {
                    let x0 = x + col as f32 * px;
                    let y1 = y - row as f32 * py;
                    push_ndc_quad(verts, x0, y1 - py, x0 + px, y1, color);
                }
            }
        }
        x += 6.0 * px;
    }
}

fn push_ndc_quad(verts: &mut Vec<UiVertex>, x0: f32, y0: f32, x1: f32, y1: f32, color: [f32; 4]) {
    let tri = [[x0, y0], [x1, y0], [x1, y1], [x0, y0], [x1, y1], [x0, y1]];
    for p in tri {
        verts.push(UiVertex { pos: p, color });
    }
}

fn model_matrix(center: [f32; 3], half: [f32; 3], q: [f32; 4]) -> [[f32; 4]; 4] {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let sx = half[0] * 2.0;
    let sy = half[1] * 2.0;
    let sz = half[2] * 2.0;
    let r00 = 1.0 - 2.0 * (y * y + z * z);
    let r01 = 2.0 * (x * y - z * w);
    let r02 = 2.0 * (x * z + y * w);
    let r10 = 2.0 * (x * y + z * w);
    let r11 = 1.0 - 2.0 * (x * x + z * z);
    let r12 = 2.0 * (y * z - x * w);
    let r20 = 2.0 * (x * z - y * w);
    let r21 = 2.0 * (y * z + x * w);
    let r22 = 1.0 - 2.0 * (x * x + y * y);
    [
        [r00 * sx, r10 * sx, r20 * sx, 0.0],
        [r01 * sy, r11 * sy, r21 * sy, 0.0],
        [r02 * sz, r12 * sz, r22 * sz, 0.0],
        [center[0], center[1], center[2], 1.0],
    ]
}

fn look_to_rh(eye: Vec3, dir: Vec3, up: Vec3) -> [[f32; 4]; 4] {
    let f = dir.normalized();
    let s = f.cross(up).normalized();
    let u = s.cross(f);
    [
        [s.x, u.x, -f.x, 0.0],
        [s.y, u.y, -f.y, 0.0],
        [s.z, u.z, -f.z, 0.0],
        [-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0],
    ]
}

fn perspective_rh(fov_y: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov_y * 0.5).tan();
    let nf = 1.0 / (near - far);
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far * nf, -1.0],
        [0.0, 0.0, near * far * nf, 0.0],
    ]
}

fn mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            out[col][row] = a[0][row] * b[col][0]
                + a[1][row] * b[col][1]
                + a[2][row] * b[col][2]
                + a[3][row] * b[col][3];
        }
    }
    out
}
