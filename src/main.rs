#[cfg(feature = "window")]
mod bitmap_font;

#[cfg(feature = "window")]
mod render;

fn main() {
    let (config, mode) = simu3d::config::Config::from_args();
    match mode {
        simu3d::config::RunMode::SelfTest => {
            let code = simu3d::self_test::run(&config);
            std::process::exit(code);
        }
        simu3d::config::RunMode::Stress => {
            let code = simu3d::self_test::stress();
            std::process::exit(code);
        }
        simu3d::config::RunMode::Window => {
            #[cfg(feature = "window")]
            {
                window::run(config);
            }
            #[cfg(not(feature = "window"))]
            {
                eprintln!("simu3d: this build has no window. Omit --no-default-features.");
                std::process::exit(2);
            }
        }
    }
}

#[cfg(feature = "window")]
mod window {
    use std::sync::Arc;
    use std::time::Instant;

    use winit::application::ApplicationHandler;
    use winit::dpi::{PhysicalPosition, PhysicalSize};
    use winit::event::{DeviceEvent, ElementState, MouseButton, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
    use winit::keyboard::{Key, KeyCode, PhysicalKey};
    #[cfg(target_os = "linux")]
    use winit::platform::x11::EventLoopBuilderExtX11;
    #[cfg(windows)]
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use winit::window::{CursorGrabMode, Window, WindowId};

    use crate::render;
    use simu3d::config::Config;
    use simu3d::level;
    use simu3d::sim::{FrameInput, Simulation};

    struct App {
        config: Config,
        window: Option<Arc<Window>>,
        gpu: Option<render::Gpu>,
        sim: Option<Simulation>,
        last: Instant,
        acc: f32,
        keys: Keys,
        grab: bool,
        yaw_counts: f64,
        pitch_counts: f64,
        raw_yaw_counts: f64,
        raw_pitch_counts: f64,
        saw_raw: bool,
        fire_edge: bool,
        looking: bool,
        /// Native X11 uses raw motion and does not warp. WSLg's raw stream is
        /// not the pointer, so that path recenters from the cursor instead.
        x11_input: bool,
        wsl: bool,
        cursor_position: Option<(f64, f64)>,
        suppress_cursor_samples: u8,
        #[cfg_attr(not(windows), allow(dead_code))]
        clip_warned: bool,
        settings_open: bool,
        setting_index: usize,
    }

    #[derive(Default)]
    struct Keys {
        w: bool,
        a: bool,
        s: bool,
        d: bool,
        shift: bool,
        jump: bool,
    }

    #[cfg(target_os = "linux")]
    fn shared_library_loads(path: &std::ffi::CStr) -> bool {
        const RTLD_NOW: i32 = 2;
        const RTLD_GLOBAL: i32 = 0x00100;
        unsafe extern "C" {
            fn dlopen(filename: *const std::ffi::c_char, flags: i32) -> *mut std::ffi::c_void;
        }
        unsafe { !dlopen(path.as_ptr(), RTLD_NOW | RTLD_GLOBAL).is_null() }
    }

    /// winit's X11 backend panics if libxkbcommon-x11.so.0 is absent. A copy in
    /// ~/.local/lib is enough: loading it here makes the later soname lookup hit
    /// the already-mapped library. The handle is kept for the process lifetime.
    #[cfg(target_os = "linux")]
    fn x11_backend_available() -> bool {
        if shared_library_loads(c"libxkbcommon-x11.so.0") {
            return true;
        }
        let Some(home) = std::env::var_os("HOME") else {
            return false;
        };
        let dir = std::path::PathBuf::from(home).join(".local/lib");
        for name in ["libxcb-xkb.so.1", "libxkbcommon-x11.so.0"] {
            let path = dir.join(name);
            let Some(text) = path.to_str() else {
                return false;
            };
            let Ok(owned) = std::ffi::CString::new(text) else {
                return false;
            };
            if !shared_library_loads(&owned) {
                return false;
            }
        }
        shared_library_loads(c"libxkbcommon-x11.so.0")
    }

    pub fn run(config: Config) {
        env_logger::init();
        let mut builder = EventLoop::builder();
        let mut x11_input = false;
        #[cfg(target_os = "linux")]
        let wsl = std::env::var_os("WSL_DISTRO_NAME").is_some();
        #[cfg(not(target_os = "linux"))]
        let wsl = false;
        #[cfg(target_os = "linux")]
        {
            let force_wayland = std::env::var_os("SIMU3D_FORCE_WAYLAND").is_some();
            let want_x11 =
                std::env::var_os("SIMU3D_FORCE_X11").is_some() || (wsl && !force_wayland);
            if want_x11 && x11_backend_available() {
                x11_input = true;
                if wsl {
                    // Before the event loop opens X. The helper is a separate
                    // client, which is what WSLg actually applies.
                    wsl_cursor::start_hider();
                    println!("input: X11 (pointer look)");
                } else {
                    println!("input: X11 (raw look, no pointer warp)");
                }
                builder.with_x11();
            } else if want_x11 {
                println!(
                    "input: Wayland. X11 look needs libxkbcommon-x11.so.0 and libxcb-xkb.so.1 (apt: libxkbcommon-x11-0 libxcb-xkb1), or those files in ~/.local/lib"
                );
            }
        }
        let event_loop = builder.build().expect("event loop");
        event_loop.set_control_flow(ControlFlow::Poll);
        let mut app = App {
            config,
            window: None,
            gpu: None,
            sim: None,
            last: Instant::now(),
            acc: 0.0,
            keys: Keys::default(),
            grab: false,
            yaw_counts: 0.0,
            pitch_counts: 0.0,
            raw_yaw_counts: 0.0,
            raw_pitch_counts: 0.0,
            saw_raw: false,
            fire_edge: false,
            looking: true,
            x11_input,
            wsl,
            cursor_position: None,
            suppress_cursor_samples: 0,
            clip_warned: false,
            settings_open: false,
            setting_index: 0,
        };
        event_loop.run_app(&mut app).expect("run");
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.window.is_some() {
                return;
            }
            let attrs = Window::default_attributes()
                .with_title(format!("simu3d — {}", level::name(self.config.level)))
                .with_inner_size(PhysicalSize::new(1600, 900));
            let window = Arc::new(event_loop.create_window(attrs).expect("window"));
            let mut gpu = pollster::block_on(render::Gpu::new(
                event_loop.owned_display_handle(),
                window.clone(),
            ));
            println!("GPU: {} ({:?})", gpu.adapter_name, gpu.backend);
            println!("Loading {}...", level::name(self.config.level));
            let mut sim = Simulation::new(self.config.clone());
            sim.reload_geometry();
            let (opaque, transparent) = sim.take_static_meshes().expect("level mesh");
            println!(
                "{}: {} blocks, {} terrain triangles",
                level::name(sim.config.level),
                sim.block_count(),
                opaque.indices.len() / 3,
            );
            let _ = transparent;
            println!(
                "Gravity {:.5} m/s². Blast splits each hit block into {} pieces. Pieces despawn after {:.1} s (simu3d.toml or --fragment-lifetime).",
                sim.config.gravity, sim.config.fragment_count, sim.config.fragment_lifetime_secs
            );
            println!(
                "ZQSD to move. Mouse movement looks while the window is focused. F1 opens settings. MAP, left and right, loads the villa, the biome, or the street. Left mouse fires. R reloads. Esc releases the mouse; click the window to capture it again. F3 prints counts."
            );
            println!("{}", level::blurb(sim.config.level));
            gpu.upload_static(&opaque, &transparent);
            self.window = Some(Arc::clone(&window));
            self.gpu = Some(gpu);
            self.sim = Some(sim);
            self.last = Instant::now();
            self.looking = true;
            self.grab_cursor(&window);
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            _id: WindowId,
            event: WindowEvent,
        ) {
            let Some(window) = self.window.clone() else {
                return;
            };
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(size) => {
                    if let Some(gpu) = &mut self.gpu {
                        gpu.resize(size);
                    }
                }
                WindowEvent::RedrawRequested => self.frame(),
                WindowEvent::KeyboardInput { event, .. } => self.on_key(event_loop, &window, event),
                WindowEvent::CursorMoved { position, .. }
                    if self.looking && !self.settings_open =>
                {
                    let pos = (position.x, position.y);
                    let size = window.inner_size();
                    let max_step = (size.width.min(size.height) as f64 * 0.25).max(64.0);
                    if let Some((dx, dy)) = take_cursor_delta(
                        &mut self.suppress_cursor_samples,
                        &mut self.cursor_position,
                        pos,
                        max_step,
                    ) {
                        self.yaw_counts += dx;
                        self.pitch_counts += dy;
                    }
                    if !self.x11_input {
                        self.pull_pointer_inside(&window);
                    }
                }
                WindowEvent::CursorLeft { .. }
                    if self.looking && !self.settings_open && !self.x11_input =>
                {
                    self.pull_pointer_inside(&window);
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    // A release does not stop look. Settings owns the pointer.
                    if state == ElementState::Pressed && !self.settings_open {
                        if !self.looking {
                            self.looking = true;
                            self.grab_cursor(&window);
                        } else if button == MouseButton::Left {
                            self.fire_edge = true;
                        }
                    }
                }
                WindowEvent::MouseWheel { delta, .. } if self.settings_open => {
                    let steps = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                        winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                    };
                    if steps.abs() > 0.01 {
                        self.nudge_setting(steps.signum());
                    }
                }
                WindowEvent::Focused(false) => self.suspend_cursor(&window),
                WindowEvent::Focused(true) if self.looking && !self.settings_open => {
                    self.grab_cursor(&window);
                }
                _ => {}
            }
        }

        fn device_event(
            &mut self,
            _event_loop: &ActiveEventLoop,
            _id: winit::event::DeviceId,
            event: DeviceEvent,
        ) {
            let Some(window) = self.window.clone() else {
                return;
            };
            if !self.look_active(&window) {
                return;
            }
            if let DeviceEvent::MouseMotion { delta } = event {
                if delta.0 != 0.0 || delta.1 != 0.0 {
                    self.saw_raw = true;
                    self.raw_yaw_counts += delta.0;
                    self.raw_pitch_counts += delta.1;
                }
            }
        }
    }

    impl App {
        fn frame(&mut self) {
            let Some(window) = self.window.clone() else {
                return;
            };
            if self.look_active(&window) {
                self.pin_pointer(&window);
            }
            let now = Instant::now();
            let frame_dt = (now - self.last).as_secs_f32().clamp(0.0, 0.05);
            self.last = now;
            let dt = self
                .sim
                .as_ref()
                .map(|sim| sim.config.fixed_dt)
                .unwrap_or(self.config.fixed_dt);
            self.acc += frame_dt;
            let (look_dx, look_dy) = if !self.looking || self.settings_open {
                (0.0, 0.0)
            } else {
                let (dx, dy) = select_look_counts(
                    (self.raw_yaw_counts, self.raw_pitch_counts),
                    (self.yaw_counts, self.pitch_counts),
                    prefer_raw_look(self.x11_input, self.wsl, cfg!(windows), self.saw_raw),
                );
                let sens = self
                    .sim
                    .as_ref()
                    .map(|sim| sim.config.mouse_sensitivity)
                    .unwrap_or(self.config.mouse_sensitivity);
                let invert = self
                    .sim
                    .as_ref()
                    .map(|sim| sim.config.invert_look)
                    .unwrap_or(self.config.invert_look);
                scale_look(dx, dy, sens, invert)
            };
            let mut look_dx = look_dx;
            let mut look_dy = look_dy;
            self.yaw_counts = 0.0;
            self.pitch_counts = 0.0;
            self.raw_yaw_counts = 0.0;
            self.raw_pitch_counts = 0.0;
            let mut steps = 0;
            while self.acc >= dt && steps < 8 {
                if let Some(sim) = &mut self.sim {
                    let input = FrameInput {
                        strafe: if self.settings_open {
                            0.0
                        } else {
                            axis(self.keys.d, self.keys.a)
                        },
                        forward: if self.settings_open {
                            0.0
                        } else {
                            axis(self.keys.w, self.keys.s)
                        },
                        jump: self.keys.jump && !self.settings_open,
                        sprint: self.keys.shift,
                        fire: self.fire_edge && !self.settings_open,
                        yaw_delta: look_dx,
                        pitch_delta: look_dy,
                    };
                    self.fire_edge = false;
                    look_dx = 0.0;
                    look_dy = 0.0;
                    sim.set_input(input);
                    sim.step(dt);
                }
                self.acc -= dt;
                steps += 1;
            }
            if let (Some(gpu), Some(sim)) = (&mut self.gpu, &mut self.sim) {
                gpu.set_block_gap(sim.config.block_gap);
                if sim.standing_gap_stale() {
                    sim.mesh_dirty = true;
                }
                if let Some((opaque, transparent)) = sim.take_static_meshes() {
                    gpu.upload_static(&opaque, &transparent);
                }
                let instances = sim.instances();
                let eye = sim.eye();
                let look = sim.look_direction();
                let aspect = gpu.aspect();
                let fov = sim.config.fov_degrees;
                let lines = if self.settings_open {
                    settings_lines(&sim.config, self.setting_index)
                } else {
                    Vec::new()
                };
                gpu.render(
                    eye,
                    look,
                    aspect,
                    &instances,
                    sim.chamber,
                    sim.reserve,
                    sim.reload_left > 0.0,
                    self.looking && !self.settings_open,
                    fov,
                    &lines,
                );
            }
            window.request_redraw();
        }

        fn on_key(
            &mut self,
            event_loop: &ActiveEventLoop,
            window: &Window,
            event: winit::event::KeyEvent,
        ) {
            let pressed = event.state == ElementState::Pressed;
            if let Some(axis) = azerty_axis(&event) {
                match axis {
                    Axis::Forward => self.keys.w = pressed,
                    Axis::Back => self.keys.s = pressed,
                    Axis::Left => self.keys.a = pressed,
                    Axis::Right => self.keys.d = pressed,
                }
                return;
            }
            match event.physical_key {
                PhysicalKey::Code(KeyCode::ShiftLeft) | PhysicalKey::Code(KeyCode::ShiftRight) => {
                    self.keys.shift = pressed
                }
                PhysicalKey::Code(KeyCode::Space) => self.keys.jump = pressed,
                PhysicalKey::Code(KeyCode::Escape) if pressed => {
                    if self.settings_open {
                        self.settings_open = false;
                        if self.looking {
                            self.grab_cursor(window);
                        }
                    } else if self.looking {
                        self.release_cursor(window);
                    } else {
                        event_loop.exit();
                    }
                }
                PhysicalKey::Code(KeyCode::F1) if pressed => {
                    self.settings_open = !self.settings_open;
                    if self.settings_open {
                        self.suspend_cursor(window);
                    } else if self.looking {
                        self.grab_cursor(window);
                    }
                }
                PhysicalKey::Code(KeyCode::ArrowUp) if pressed && self.settings_open => {
                    self.setting_index = self.setting_index.saturating_sub(1);
                }
                PhysicalKey::Code(KeyCode::ArrowDown) if pressed && self.settings_open => {
                    self.setting_index = (self.setting_index + 1).min(KNOBS.len() + 1);
                }
                PhysicalKey::Code(KeyCode::ArrowLeft) if pressed && self.settings_open => {
                    self.nudge_setting(-1.0);
                }
                PhysicalKey::Code(KeyCode::ArrowRight) if pressed && self.settings_open => {
                    self.nudge_setting(1.0);
                }
                PhysicalKey::Code(KeyCode::KeyR) if pressed => {
                    if let Some(sim) = &mut self.sim {
                        if sim.chamber == 0 && sim.reload_left <= 0.0 {
                            if sim.reserve == 0 {
                                sim.reserve = sim.config.rocket_reserve.max(1);
                            }
                            sim.reload_left = sim.config.rocket_reload_secs;
                        }
                    }
                }
                PhysicalKey::Code(KeyCode::Enter) if pressed && self.settings_open => {
                    if self.setting_index == KNOBS.len() + 1 {
                        self.reload_map();
                    }
                }
                PhysicalKey::Code(KeyCode::F3) if pressed => {
                    if let Some(sim) = &self.sim {
                        println!("{}", sim.status_line());
                    }
                }
                _ => {}
            }
        }

        fn grab_cursor(&mut self, window: &Window) {
            self.grab = true;
            self.saw_raw = false;
            self.cursor_position = None;
            self.suppress_cursor_samples = 2;
            self.yaw_counts = 0.0;
            self.pitch_counts = 0.0;
            self.raw_yaw_counts = 0.0;
            self.raw_pitch_counts = 0.0;
            if !self.wsl {
                window.set_cursor_visible(false);
            }
            if let Err(err) = grab_platform_cursor(window) {
                eprintln!("simu3d: cursor grab failed ({err}); look still follows the pointer while it is inside the window");
            }
            // An empty X pixmap is what WSLg draws as the normal arrow. The
            // Windows pointer is already blank, so that arrow is the one left
            // inside the window.
            self.conceal_pointer(window);
            self.pin_pointer(window);
        }

        fn look_active(&self, window: &Window) -> bool {
            self.looking && !self.settings_open && window.has_focus()
        }

        fn release_cursor(&mut self, window: &Window) {
            self.looking = false;
            self.suspend_cursor(window);
        }

        /// Drop the OS grab without forgetting that mouse look is the normal mode.
        /// Focus loss and the settings panel use this. Esc uses `release_cursor`.
        fn suspend_cursor(&mut self, window: &Window) {
            self.grab = false;
            self.saw_raw = false;
            self.cursor_position = None;
            self.suppress_cursor_samples = 0;
            self.yaw_counts = 0.0;
            self.pitch_counts = 0.0;
            self.raw_yaw_counts = 0.0;
            self.raw_pitch_counts = 0.0;
            #[cfg(windows)]
            unclip_cursor();
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            self.reveal_pointer(window);
            window.set_cursor_visible(true);
        }

        fn conceal_pointer(&mut self, window: &Window) {
            #[cfg(target_os = "linux")]
            if self.wsl {
                wsl_cursor::conceal(window);
            }
            #[cfg(not(target_os = "linux"))]
            let _ = window;
        }

        fn reveal_pointer(&mut self, window: &Window) {
            #[cfg(target_os = "linux")]
            if self.wsl {
                wsl_cursor::reveal(window);
            }
            #[cfg(not(target_os = "linux"))]
            let _ = window;
        }

        /// Re-apply confinement every frame. winit only clips while the window is
        /// the active window at the moment grab is requested, and Windows drops
        /// that clip when focus flickers. The OS pointer then walks out of the
        /// client area and look events stop.
        fn pin_pointer(&mut self, window: &Window) {
            if !window.has_focus() {
                return;
            }
            if self.looking && !self.settings_open {
                if !self.wsl {
                    window.set_cursor_visible(false);
                }
                self.conceal_pointer(window);
            }
            if !self.grab && grab_platform_cursor(window).is_ok() {
                self.grab = true;
            }
            #[cfg(windows)]
            if !clip_cursor_to_client(window) && !self.clip_warned {
                self.clip_warned = true;
                eprintln!("simu3d: ClipCursor failed; the pointer can leave the window");
            }
            // Native X11 raw motion does not need the pointer put back.
            // WSLg's raw stream is not that pointer: cursor pixels are. Recenter
            // only near the edges. Warping every sample is the WSLg round trip
            // that makes look lag.
            if !self.x11_input || self.wsl {
                self.pull_pointer_inside(window);
            }
        }

        fn pull_pointer_inside(&mut self, window: &Window) {
            if !self.looking || self.settings_open || !window.has_focus() {
                return;
            }
            #[cfg(windows)]
            {
                let _ = clip_cursor_to_client(window);
            }
            let size = window.inner_size();
            if !cursor_needs_recenter(self.cursor_position, size) {
                return;
            }
            if recenter_cursor(window) {
                self.cursor_position = Some((size.width as f64 * 0.5, size.height as f64 * 0.5));
                self.suppress_cursor_samples = self.suppress_cursor_samples.max(1);
            }
        }

        fn nudge_setting(&mut self, steps: f32) {
            if self.setting_index == KNOBS.len() {
                self.cycle_map(steps.signum());
                return;
            }
            if self.setting_index == KNOBS.len() + 1 {
                if steps > 0.0 {
                    self.reload_map();
                }
                return;
            }
            let knob = &KNOBS[self.setting_index];
            let Some(sim) = &mut self.sim else {
                return;
            };
            let current = (knob.get)(&sim.config);
            let value = step_knob(knob.label, current, steps, knob.step, knob.min, knob.max);
            (knob.set)(&mut sim.config, value);
            self.config = sim.config.clone();
            if let Err(err) = self.config.save_default() {
                eprintln!("simu3d: could not write simu3d.toml: {err}");
            }
        }

        fn cycle_map(&mut self, dir: f32) {
            let Some(current) = self.sim.as_ref().map(|sim| sim.config.level) else {
                return;
            };
            let step = if dir < 0.0 { -1i32 } else { 1 };
            self.config.level = (current as i32 + step).rem_euclid(level::COUNT as i32) as u8;
            if let Err(err) = self.config.save_default() {
                eprintln!("simu3d: could not write simu3d.toml: {err}");
            }
            self.reload_map();
        }

        fn reload_map(&mut self) {
            let title = {
                let Some(sim) = &mut self.sim else {
                    return;
                };
                sim.config.level = self.config.level;
                let name = level::name(sim.config.level);
                println!("loading {name}...");
                sim.reload_geometry();
                println!("loaded {name} — {} blocks", sim.block_count());
                println!("{}", level::blurb(sim.config.level));
                format!("simu3d — {name}")
            };
            if let Some(window) = &self.window {
                window.set_title(&title);
            }
        }
    }

    fn axis(positive: bool, negative: bool) -> f32 {
        let mut v = 0.0;
        if positive {
            v += 1.0;
        }
        if negative {
            v -= 1.0;
        }
        v
    }

    struct Knob {
        label: &'static str,
        min: f32,
        max: f32,
        step: f32,
        get: fn(&Config) -> f32,
        set: fn(&mut Config, f32),
    }

    const KNOBS: &[Knob] = &[
        Knob {
            label: "FOV",
            min: 40.0,
            max: 110.0,
            step: 1.0,
            get: |c| c.fov_degrees,
            set: |c, v| c.fov_degrees = v,
        },
        Knob {
            label: "INVERT",
            min: 0.0,
            max: 1.0,
            step: 1.0,
            get: |c| c.invert_look,
            set: |c, v| c.invert_look = if v >= 0.5 { 1.0 } else { 0.0 },
        },
        Knob {
            label: "GRAVITY",
            min: 0.5,
            max: 30.0,
            step: 0.1,
            get: |c| c.gravity,
            set: |c, v| c.gravity = v,
        },
        Knob {
            label: "LIFETIME",
            min: 0.5,
            max: 30.0,
            step: 0.5,
            get: |c| c.fragment_lifetime_secs,
            set: |c, v| c.fragment_lifetime_secs = v,
        },
        Knob {
            label: "RADIUS",
            min: 0.15,
            max: 2.0,
            step: 0.05,
            get: |c| c.explosion_radius,
            set: |c, v| c.explosion_radius = v,
        },
        Knob {
            label: "IMPULSE",
            min: 200.0,
            max: 20000.0,
            step: 100.0,
            get: |c| c.blast_impulse,
            set: |c, v| c.blast_impulse = v,
        },
        Knob {
            label: "STONE ADH",
            min: 20.0,
            max: 400.0,
            step: 5.0,
            get: |c| c.adhesive_stone,
            set: |c, v| c.adhesive_stone = v,
        },
        Knob {
            label: "GLASS ADH",
            min: 0.0,
            max: 40.0,
            step: 1.0,
            get: |c| c.adhesive_glass,
            set: |c, v| c.adhesive_glass = v,
        },
        Knob {
            label: "ROCKET",
            min: 10.0,
            max: 200.0,
            step: 1.0,
            get: |c| c.rocket_speed,
            set: |c, v| c.rocket_speed = v,
        },
        Knob {
            label: "RELOAD",
            min: 0.2,
            max: 5.0,
            step: 0.05,
            get: |c| c.rocket_reload_secs,
            set: |c, v| c.rocket_reload_secs = v,
        },
        Knob {
            label: "RESERVE",
            min: 0.0,
            max: 30.0,
            step: 1.0,
            get: |c| c.rocket_reserve as f32,
            set: |c, v| c.rocket_reserve = v.round().max(0.0) as u32,
        },
        Knob {
            label: "SENS",
            min: SENS_MIN,
            max: SENS_MAX,
            step: 0.0002,
            get: |c| c.mouse_sensitivity,
            set: |c, v| c.mouse_sensitivity = v,
        },
        Knob {
            label: "WALK",
            min: 1.0,
            max: 12.0,
            step: 0.1,
            get: |c| c.walk_speed,
            set: |c, v| c.walk_speed = v,
        },
        Knob {
            label: "SPRINT",
            min: 2.0,
            max: 20.0,
            step: 0.1,
            get: |c| c.sprint_speed,
            set: |c, v| c.sprint_speed = v,
        },
        Knob {
            label: "JUMP",
            min: 1.0,
            max: 12.0,
            step: 0.1,
            get: |c| c.jump_speed,
            set: |c, v| c.jump_speed = v,
        },
        Knob {
            label: "HEIGHT",
            min: 0.8,
            max: 3.0,
            step: 0.05,
            get: |c| c.player_height_m,
            set: |c, v| c.player_height_m = v,
        },
        Knob {
            label: "DT",
            min: 1.0 / 240.0,
            max: 1.0 / 30.0,
            step: 0.001,
            get: |c| c.fixed_dt,
            set: |c, v| c.fixed_dt = v,
        },
        Knob {
            label: "BOX",
            min: 0.1,
            max: 0.5,
            step: 0.05,
            get: |c| c.block_size,
            set: |c, v| c.block_size = v,
        },
        Knob {
            label: "JOINT",
            min: 0.0,
            max: 0.16,
            step: 0.004,
            get: |c| c.block_gap,
            set: |c, v| c.block_gap = v,
        },
    ];

    fn settings_lines(config: &Config, selected: usize) -> Vec<render::SettingLine> {
        let mut lines: Vec<render::SettingLine> = KNOBS
            .iter()
            .enumerate()
            .map(|(index, knob)| {
                let value = (knob.get)(config);
                let text = if knob.label == "INVERT" {
                    format!("INVERT {}", if value >= 0.5 { "ON" } else { "OFF" })
                } else if knob.label == "SENS" {
                    format!("SENS {value:.5}")
                } else {
                    format!("{} {:.4}", knob.label, value)
                };
                render::SettingLine {
                    text,
                    selected: index == selected,
                }
            })
            .collect();
        lines.push(render::SettingLine {
            text: format!("MAP {}", level::name(config.level)),
            selected: selected == KNOBS.len(),
        });
        lines.push(render::SettingLine {
            text: "RELOAD MAP".to_string(),
            selected: selected == KNOBS.len() + 1,
        });
        lines
    }

    enum Axis {
        Forward,
        Back,
        Left,
        Right,
    }

    /// Letters on an AZERTY keyboard: Z forward, Q left, S back, D right.
    /// W and A, the QWERTY letters, do not move.
    /// If the platform reports no character, the physical W and A positions are used,
    /// because those positions are labelled Z and Q on an AZERTY board.
    fn azerty_axis(event: &winit::event::KeyEvent) -> Option<Axis> {
        if let Key::Character(text) = &event.logical_key {
            match text.as_str() {
                "z" | "Z" => return Some(Axis::Forward),
                "q" | "Q" => return Some(Axis::Left),
                "s" | "S" => return Some(Axis::Back),
                "d" | "D" => return Some(Axis::Right),
                "w" | "W" | "a" | "A" => return None,
                _ => {}
            }
        }
        match event.physical_key {
            PhysicalKey::Code(KeyCode::KeyW) => Some(Axis::Forward),
            PhysicalKey::Code(KeyCode::KeyA) => Some(Axis::Left),
            PhysicalKey::Code(KeyCode::KeyS) => Some(Axis::Back),
            PhysicalKey::Code(KeyCode::KeyD) => Some(Axis::Right),
            _ => None,
        }
    }

    const SENS_MIN: f32 = 0.0001;
    const SENS_MAX: f32 = 0.05;

    fn take_cursor_delta(
        suppress: &mut u8,
        previous: &mut Option<(f64, f64)>,
        current: (f64, f64),
        max_step: f64,
    ) -> Option<(f64, f64)> {
        if *suppress > 0 {
            *suppress -= 1;
            *previous = Some(current);
            return None;
        }
        let delta = previous.map(|prev| (current.0 - prev.0, current.1 - prev.1));
        *previous = Some(current);
        // A recenter is a jump from the edge to the middle. That is not a mouse count.
        match delta {
            Some((dx, dy)) if dx.abs() > max_step || dy.abs() > max_step => None,
            other => other,
        }
    }

    /// Native X11 raw motion is relative and continuous. Prefer it, and do not warp.
    /// WSLg must not: its XI2 raw steps are thousands of counts while the pointer
    /// moves a few pixels, and preferring them flings the view off the mouse.
    /// Wayland must not either: one raw event, then a dead stream.
    fn prefer_raw_look(x11: bool, wsl: bool, windows: bool, saw_raw: bool) -> bool {
        (x11 && !wsl) || (windows && saw_raw)
    }

    /// When raw is trusted, a warp's cursor sample would cancel it, so raw wins.
    /// Otherwise only cursor pixels look. An idle cursor does not fall back to
    /// raw: on WSLg that fallback is a stuck pitch and a spinning yaw.
    fn select_look_counts(raw: (f64, f64), cursor: (f64, f64), prefer_raw: bool) -> (f64, f64) {
        let raw_moved = raw.0 != 0.0 || raw.1 != 0.0;
        let cursor_moved = cursor.0 != 0.0 || cursor.1 != 0.0;
        if prefer_raw && raw_moved {
            raw
        } else if cursor_moved {
            cursor
        } else if prefer_raw {
            raw
        } else {
            (0.0, 0.0)
        }
    }

    fn scale_look(dx: f64, dy: f64, sensitivity: f32, invert_look: f32) -> (f32, f32) {
        let pitch_sign = if invert_look >= 0.5 { -1.0 } else { 1.0 };
        (
            dx as f32 * sensitivity,
            dy as f32 * sensitivity * pitch_sign,
        )
    }

    fn cursor_needs_recenter(position: Option<(f64, f64)>, size: PhysicalSize<u32>) -> bool {
        let Some((x, y)) = position else {
            return true;
        };
        if size.width < 2 || size.height < 2 {
            return false;
        }
        // Stay put in the middle. A warp on every pixel is a WSLg round trip.
        let margin_x = (size.width as f64 * 0.2).max(48.0);
        let margin_y = (size.height as f64 * 0.2).max(48.0);
        x < margin_x
            || y < margin_y
            || x > size.width as f64 - margin_x
            || y > size.height as f64 - margin_y
    }

    fn recenter_cursor(window: &Window) -> bool {
        let size = window.inner_size();
        if size.width < 2 || size.height < 2 {
            return false;
        }
        let center = PhysicalPosition::new(size.width as f64 * 0.5, size.height as f64 * 0.5);
        window.set_cursor_position(center).is_ok()
    }

    #[cfg(target_os = "linux")]
    mod wsl_cursor {
        use std::ffi::{c_char, c_void};
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::OnceLock;

        use winit::window::Window;

        #[link(name = "X11")]
        unsafe extern "C" {
            fn XOpenDisplay(name: *const c_char) -> *mut c_void;
            fn XDefaultRootWindow(display: *mut c_void) -> u64;
            fn XFlush(display: *mut c_void) -> i32;
            fn XSync(display: *mut c_void, discard: i32) -> i32;
            fn XDefineCursor(display: *mut c_void, window: u64, cursor: u64) -> i32;
            fn XFetchName(display: *mut c_void, window: u64, name: *mut *mut c_char) -> i32;
            fn XInternAtom(display: *mut c_void, name: *const c_char, only_if_exists: i32) -> u64;
            fn XGetWindowProperty(
                display: *mut c_void,
                window: u64,
                property: u64,
                long_offset: i64,
                long_length: i64,
                delete: i32,
                req_type: u64,
                actual_type: *mut u64,
                actual_format: *mut i32,
                nitems: *mut u64,
                bytes_after: *mut u64,
                prop: *mut *mut u8,
            ) -> i32;
            fn XQueryTree(
                display: *mut c_void,
                window: u64,
                root: *mut u64,
                parent: *mut u64,
                children: *mut *mut u64,
                nchildren: *mut u32,
            ) -> i32;
            fn XFree(data: *mut c_void) -> i32;
        }

        #[link(name = "dl")]
        unsafe extern "C" {
            fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
            fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        }

        /// ARGB cursor image. Field order matches libXcursor.
        #[repr(C)]
        struct XcursorImage {
            version: u32,
            size: u32,
            width: u32,
            height: u32,
            xhot: u32,
            yhot: u32,
            delay: u32,
            pixels: *mut u32,
        }

        struct XcursorFns {
            create: unsafe extern "C" fn(i32, i32) -> *mut XcursorImage,
            destroy: unsafe extern "C" fn(*mut XcursorImage),
            load: unsafe extern "C" fn(*mut c_void, *const XcursorImage) -> u64,
        }

        fn xcursor() -> Option<&'static XcursorFns> {
            static ONCE: OnceLock<Option<XcursorFns>> = OnceLock::new();
            ONCE.get_or_init(|| unsafe {
                let handle = dlopen(b"libXcursor.so.1\0".as_ptr().cast(), 1);
                if handle.is_null() {
                    return None;
                }
                let create = dlsym(handle, b"XcursorImageCreate\0".as_ptr().cast());
                let destroy = dlsym(handle, b"XcursorImageDestroy\0".as_ptr().cast());
                let load = dlsym(handle, b"XcursorImageLoadCursor\0".as_ptr().cast());
                if create.is_null() || destroy.is_null() || load.is_null() {
                    return None;
                }
                Some(XcursorFns {
                    create: std::mem::transmute(create),
                    destroy: std::mem::transmute(destroy),
                    load: std::mem::transmute(load),
                })
            })
            .as_ref()
        }

        /// A real ARGB cursor. An empty 1-bit pixmap is replaced by WSLg with
        /// the theme arrow, which is the pointer that stays inside the window.
        fn transparent_cursor(display: *mut c_void) -> u64 {
            static CURSOR: AtomicU64 = AtomicU64::new(0);
            let existing = CURSOR.load(Ordering::Relaxed);
            if existing != 0 {
                return existing;
            }
            let Some(fns) = xcursor() else {
                return 0;
            };
            unsafe {
                let image = (fns.create)(32, 32);
                if image.is_null() {
                    return 0;
                }
                let count = ((*image).width as usize).saturating_mul((*image).height as usize);
                let pixels = (*image).pixels;
                if !pixels.is_null() {
                    for i in 0..count {
                        // Alpha 0 is dropped by WSLg, which keeps the arrow.
                        // An image whose RGB is all zero is also dropped when a
                        // cursor is already showing. One blue channel of 1 makes
                        // the update stick, and alpha 1 does not paint.
                        *pixels.add(i) = 0x0100_0000;
                    }
                    if count > 0 {
                        *pixels = 0x0100_0001;
                    }
                }
                let cursor = (fns.load)(display, image);
                (fns.destroy)(image);
                if cursor != 0 {
                    CURSOR.store(cursor, Ordering::Relaxed);
                }
                cursor
            }
        }

        fn server() -> *mut c_void {
            static DISPLAY: AtomicU64 = AtomicU64::new(0);
            let existing = DISPLAY.load(Ordering::Relaxed);
            if existing != 0 {
                return existing as *mut c_void;
            }
            let opened = unsafe { XOpenDisplay(std::ptr::null()) };
            if opened.is_null() {
                return std::ptr::null_mut();
            }
            match DISPLAY.compare_exchange(0, opened as u64, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => opened,
                Err(prev) => prev as *mut c_void,
            }
        }

        fn title_is_simu(display: *mut c_void, window: u64) -> bool {
            unsafe {
                let mut name: *mut c_char = std::ptr::null_mut();
                if XFetchName(display, window, &mut name) != 0 && !name.is_null() {
                    let ok = std::ffi::CStr::from_ptr(name)
                        .to_bytes()
                        .windows(6)
                        .any(|part| part == b"simu3d");
                    XFree(name.cast());
                    if ok {
                        return true;
                    }
                }
                let net = XInternAtom(display, b"_NET_WM_NAME\0".as_ptr().cast(), 0);
                let mut actual = 0u64;
                let mut format = 0i32;
                let mut count = 0u64;
                let mut after = 0u64;
                let mut data: *mut u8 = std::ptr::null_mut();
                // AnyPropertyType is 0. WM_NAME above misses a UTF-8-only title.
                let rc = XGetWindowProperty(
                    display,
                    window,
                    net,
                    0,
                    64,
                    0,
                    0,
                    &mut actual,
                    &mut format,
                    &mut count,
                    &mut after,
                    &mut data,
                );
                if rc == 0 && !data.is_null() && count > 0 {
                    let bytes = std::slice::from_raw_parts(data, count as usize);
                    let ok = bytes.windows(6).any(|part| part == b"simu3d");
                    XFree(data.cast());
                    return ok;
                }
                if !data.is_null() {
                    XFree(data.cast());
                }
                false
            }
        }

        fn define_tree(
            display: *mut c_void,
            window: u64,
            cursor: u64,
            depth: i32,
            hits: &mut Vec<u64>,
        ) {
            if depth > 12 || window == 0 {
                return;
            }
            if title_is_simu(display, window) {
                unsafe { XDefineCursor(display, window, cursor) };
                hits.push(window);
            }
            unsafe {
                let mut root = 0u64;
                let mut parent = 0u64;
                let mut kids: *mut u64 = std::ptr::null_mut();
                let mut n = 0u32;
                if XQueryTree(display, window, &mut root, &mut parent, &mut kids, &mut n) == 0 {
                    return;
                }
                if !kids.is_null() {
                    let children = std::slice::from_raw_parts(kids, n as usize);
                    for &child in children {
                        define_tree(display, child, cursor, depth + 1, hits);
                    }
                    XFree(kids.cast());
                }
            }
        }

        unsafe extern "C" {
            fn fork() -> i32;
            fn getppid() -> i32;
            fn _exit(code: i32) -> !;
            // PR_SET_PDEATHSIG = 1. The helper exits when the game does.
            fn prctl(option: i32, arg2: u64, arg3: u64, arg4: u64, arg5: u64) -> i32;
        }

        /// WSLg keeps the theme arrow when the game defines the blank cursor
        /// on winit's display. A child started before that display opens is
        /// the client whose cursor replaces the arrow.
        pub fn start_hider() {
            let pid = unsafe { fork() };
            if pid != 0 {
                if pid < 0 {
                    eprintln!("simu3d: pointer hide could not start");
                }
                return;
            }
            unsafe {
                let parent = getppid();
                prctl(1, 15, 0, 0, 0);
                if getppid() != parent {
                    _exit(0);
                }
            }
            let display = server();
            if display.is_null() {
                eprintln!("simu3d: pointer hide could not open X");
                unsafe { _exit(1) };
            }
            let cursor = transparent_cursor(display);
            if cursor == 0 {
                eprintln!("simu3d: pointer hide could not build a cursor");
                unsafe { _exit(1) };
            }
            let mut logged = false;
            loop {
                if unsafe { getppid() } <= 1 {
                    break;
                }
                let root = unsafe { XDefaultRootWindow(display) };
                let mut hits = Vec::new();
                define_tree(display, root, cursor, 0, &mut hits);
                if !logged && !hits.is_empty() {
                    logged = true;
                    for window in &hits {
                        eprintln!("simu3d: hiding pointer on xwindow {window:#x}");
                    }
                    unsafe { XSync(display, 0) };
                }
                unsafe { XFlush(display) };
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
            unsafe { _exit(0) };
        }

        pub fn conceal(_window: &Window) {}

        pub fn reveal(_window: &Window) {}
    }

    fn grab_platform_cursor(window: &Window) -> Result<(), winit::error::ExternalError> {
        // Windows Locked is a 1px ClipCursor. It does not hold on this machine:
        // the OS pointer keeps moving and look stops when it leaves the window.
        // Confined is the client rectangle. The frame loop re-applies it.
        #[cfg(windows)]
        {
            window
                .set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Locked))
        }
        #[cfg(not(windows))]
        {
            // Locked on Wayland hides the pointer and stops CursorMoved, and
            // WSLg does not deliver the relative stream that lock is supposed
            // to replace it with. Confined keeps motion events. Warp recentres.
            window
                .set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Locked))
        }
    }

    /// Sensitivity uses a multiplicative step so the panel can cover 0.0001 to 0.05
    /// without a hundred wheel clicks. Other knobs stay additive.
    fn step_knob(label: &str, current: f32, steps: f32, step: f32, min: f32, max: f32) -> f32 {
        let value = if label == "SENS" {
            current * 1.5_f32.powf(steps)
        } else {
            current + steps * step
        };
        value.clamp(min, max)
    }

    #[cfg(windows)]
    mod win_cursor {
        #[repr(C)]
        #[derive(Clone, Copy, PartialEq, Eq)]
        struct Rect {
            left: i32,
            top: i32,
            right: i32,
            bottom: i32,
        }

        #[repr(C)]
        struct Point {
            x: i32,
            y: i32,
        }

        #[link(name = "user32")]
        extern "system" {
            fn GetClientRect(hwnd: *mut std::ffi::c_void, rect: *mut Rect) -> i32;
            fn ClientToScreen(hwnd: *mut std::ffi::c_void, point: *mut Point) -> i32;
            fn ClipCursor(rect: *const Rect) -> i32;
            fn GetClipCursor(rect: *mut Rect) -> i32;
        }

        pub fn clip_to_client(hwnd: *mut std::ffi::c_void) -> bool {
            unsafe {
                let mut client = Rect {
                    left: 0,
                    top: 0,
                    right: 0,
                    bottom: 0,
                };
                if GetClientRect(hwnd, &mut client) == 0 {
                    return false;
                }
                let mut origin = Point { x: 0, y: 0 };
                if ClientToScreen(hwnd, &mut origin) == 0 {
                    return false;
                }
                let desired = Rect {
                    left: origin.x + client.left,
                    top: origin.y + client.top,
                    right: origin.x + client.right,
                    bottom: origin.y + client.bottom,
                };
                if desired.right <= desired.left || desired.bottom <= desired.top {
                    return false;
                }
                let mut current = Rect {
                    left: 0,
                    top: 0,
                    right: 0,
                    bottom: 0,
                };
                if GetClipCursor(&mut current) != 0 && current == desired {
                    return true;
                }
                ClipCursor(&desired) != 0
            }
        }

        pub fn unclip() {
            unsafe {
                ClipCursor(std::ptr::null());
            }
        }
    }

    #[cfg(windows)]
    fn window_hwnd(window: &Window) -> Option<*mut std::ffi::c_void> {
        let handle = window.window_handle().ok()?;
        match handle.as_raw() {
            RawWindowHandle::Win32(win) => Some(win.hwnd.get() as *mut std::ffi::c_void),
            _ => None,
        }
    }

    #[cfg(windows)]
    fn clip_cursor_to_client(window: &Window) -> bool {
        let Some(hwnd) = window_hwnd(window) else {
            return false;
        };
        win_cursor::clip_to_client(hwnd)
    }

    #[cfg(windows)]
    fn unclip_cursor() {
        win_cursor::unclip();
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn warp_sample_is_not_a_look_delta() {
            let mut previous = None;
            let mut suppress = 1u8;
            assert_eq!(
                take_cursor_delta(&mut suppress, &mut previous, (800.0, 450.0), 400.0),
                None
            );
            assert_eq!(suppress, 0);
            assert_eq!(
                take_cursor_delta(&mut suppress, &mut previous, (820.0, 440.0), 400.0),
                Some((20.0, -10.0))
            );
            let mut at_edge = Some((1500.0, 450.0));
            assert_eq!(
                take_cursor_delta(&mut suppress, &mut at_edge, (800.0, 450.0), 400.0),
                None
            );
            assert_eq!(
                take_cursor_delta(&mut suppress, &mut at_edge, (830.0, 440.0), 400.0),
                Some((30.0, -10.0))
            );
        }

        #[test]
        fn raw_motion_is_not_mixed_with_the_recenter_warp() {
            let raw = (12.0, -4.0);
            let warp = (-12.0, 4.0);
            assert_eq!(select_look_counts(raw, warp, true), raw);
            assert!(prefer_raw_look(true, false, false, true));
            assert!(!prefer_raw_look(true, true, false, true));
            assert!(!prefer_raw_look(false, false, false, true));
            assert_eq!(select_look_counts(raw, (0.0, 0.0), true), raw);
            assert_eq!(select_look_counts(raw, (0.0, 0.0), false), (0.0, 0.0));
            assert_eq!(select_look_counts(raw, warp, false), warp);
            // Measured on WSLg: raw steps of ~18000 while the pointer moved 14 px.
            assert_eq!(
                select_look_counts((18028.0, 430.0), (14.0, 0.0), false),
                (14.0, 0.0)
            );
        }

        #[test]
        fn sensitivity_panel_reaches_past_the_old_ceiling() {
            assert!(SENS_MAX > 0.01);
            assert!(SENS_MIN < 0.0004);
            let up = step_knob("SENS", 0.0022, 1.0, 0.0002, SENS_MIN, SENS_MAX);
            assert!((up - 0.0033).abs() < 1.0e-6);
            assert_eq!(
                step_knob("SENS", 0.04, 1.0, 0.0002, SENS_MIN, SENS_MAX),
                SENS_MAX
            );
            assert_eq!(
                step_knob("SENS", 0.00012, -1.0, 0.0002, SENS_MIN, SENS_MAX),
                SENS_MIN
            );
            assert_eq!(step_knob("WALK", 4.3, 1.0, 0.1, 1.0, 12.0), 4.4);
        }

        #[test]
        fn recenter_when_the_pointer_nears_the_edge() {
            let size = PhysicalSize::new(1600, 900);
            assert!(cursor_needs_recenter(None, size));
            assert!(!cursor_needs_recenter(Some((800.0, 450.0)), size));
            assert!(!cursor_needs_recenter(Some((1000.0, 450.0)), size));
            assert!(cursor_needs_recenter(Some((1500.0, 450.0)), size));
        }

        #[test]
        fn invert_look_flips_only_pitch() {
            let (yaw, pitch) = scale_look(10.0, 4.0, 0.002, 1.0);
            assert!((yaw - 0.02).abs() < 1.0e-6, "yaw {yaw}");
            assert!((pitch + 0.008).abs() < 1.0e-6, "pitch {pitch}");
            let (yaw, pitch) = scale_look(10.0, 4.0, 0.002, 0.0);
            assert!((yaw - 0.02).abs() < 1.0e-6);
            assert!((pitch - 0.008).abs() < 1.0e-6);
        }
    }
}
