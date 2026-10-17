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
            simu3d::self_test::stress();
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
    #[cfg(windows)]
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use winit::event::{DeviceEvent, ElementState, MouseButton, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
    use winit::keyboard::{Key, KeyCode, PhysicalKey};
    use winit::window::{CursorGrabMode, Window, WindowId};
    #[cfg(target_os = "linux")]
    use winit::platform::x11::EventLoopBuilderExtX11;

    use simu3d::config::Config;
    use simu3d::sim::{FrameInput, Simulation};
    use simu3d::villa;
    use crate::render;

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

    pub fn run(config: Config) {
        env_logger::init();
        let mut builder = EventLoop::builder();
        #[cfg(target_os = "linux")]
        if std::env::var_os("SIMU3D_FORCE_X11").is_some() {
            builder.with_x11();
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
            looking: false,
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
                .with_title("simu3d — villa")
                .with_inner_size(PhysicalSize::new(1600, 900));
            let window = Arc::new(event_loop.create_window(attrs).expect("window"));
            let mut gpu = pollster::block_on(render::Gpu::new(
                event_loop.owned_display_handle(),
                window.clone(),
            ));
            println!("GPU: {} ({:?})", gpu.adapter_name, gpu.backend);
            println!("Building villa...");
            let mut sim = Simulation::new(self.config.clone());
            let spawn = villa::build(&mut sim.voxels);
            sim.place_player(spawn.feet, spawn.yaw, spawn.pitch);
            sim.refresh_static();
            let (opaque, transparent) = sim
                .take_static_meshes()
                .expect("villa mesh");
            println!(
                "Villa: {} blocks, {} opaque triangles, {} glass triangles",
                sim.block_count(),
                opaque.indices.len() / 3,
                transparent.indices.len() / 3,
            );
            println!(
                "Gravity {:.5} m/s². Blast splits each hit block into {} pieces. Pieces despawn after {:.1} s (simu3d.toml or --fragment-lifetime).",
                sim.config.gravity, sim.config.fragment_count, sim.config.fragment_lifetime_secs
            );
            println!(
                "ZQSD to move. Hold the right mouse button to look. The pointer is pinned inside the window while you look. F1 opens settings: field of view and every simulation value. Left mouse fires. R reloads. Esc releases the mouse. F3 prints counts."
            );
            println!(
                "The metal fence does not break. The ground does: blow a hole and you fall through it. There is no floor under the plot."
            );
            gpu.upload_static(&opaque, &transparent);
            self.window = Some(window);
            self.gpu = Some(gpu);
            self.sim = Some(sim);
            self.last = Instant::now();
        }

        fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
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
                    if let Some((dx, dy)) = take_cursor_delta(
                        &mut self.suppress_cursor_samples,
                        &mut self.cursor_position,
                        pos,
                    ) {
                        self.yaw_counts += dx;
                        self.pitch_counts += dy;
                    }
                    self.pull_pointer_inside(&window);
                }
                WindowEvent::CursorLeft { .. } if self.looking && !self.settings_open => {
                    self.pull_pointer_inside(&window);
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    let pressed = state == ElementState::Pressed;
                    match button {
                        MouseButton::Right => {
                            self.looking = pressed && !self.settings_open;
                            if self.looking {
                                self.grab_cursor(&window);
                            } else {
                                self.release_cursor(&window);
                            }
                        }
                        MouseButton::Left if pressed && !self.settings_open => {
                            self.fire_edge = true;
                        }
                        _ => {}
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
                WindowEvent::Focused(false) => self.release_cursor(&window),
                _ => {}
            }
        }

        fn device_event(
            &mut self,
            _event_loop: &ActiveEventLoop,
            _id: winit::event::DeviceId,
            event: DeviceEvent,
        ) {
            if !self.looking || self.settings_open {
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
            if self.looking && !self.settings_open {
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
            let (look_dx, look_dy) = if self.settings_open || !self.looking {
                (0.0, 0.0)
            } else {
                let (dx, dy) = select_look_counts(
                    (self.raw_yaw_counts, self.raw_pitch_counts),
                    (self.yaw_counts, self.pitch_counts),
                    self.saw_raw,
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
                    !self.looking,
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
                    } else if self.grab || self.looking {
                        self.looking = false;
                        self.release_cursor(window);
                    } else {
                        event_loop.exit();
                    }
                }
                PhysicalKey::Code(KeyCode::F1) if pressed => {
                    self.settings_open = !self.settings_open;
                    self.looking = false;
                    self.release_cursor(window);
                }
                PhysicalKey::Code(KeyCode::ArrowUp) if pressed && self.settings_open => {
                    self.setting_index = self.setting_index.saturating_sub(1);
                }
                PhysicalKey::Code(KeyCode::ArrowDown) if pressed && self.settings_open => {
                    self.setting_index = (self.setting_index + 1).min(KNOBS.len());
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
                    if self.setting_index == KNOBS.len() {
                        self.reload_house();
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
            window.set_cursor_visible(false);
            if let Err(err) = grab_platform_cursor(window) {
                eprintln!("simu3d: cursor grab failed ({err}); pinning the pointer inside the window");
            }
            self.pin_pointer(window);
        }

        fn release_cursor(&mut self, window: &Window) {
            self.grab = false;
            self.looking = false;
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
            window.set_cursor_visible(true);
        }

        /// Re-apply confinement every frame. winit only clips while the window is
        /// the active window at the moment grab is requested, and Windows drops
        /// that clip when focus flickers. The OS pointer then walks out of the
        /// client area and look events stop.
        fn pin_pointer(&mut self, window: &Window) {
            if !window.has_focus() {
                return;
            }
            window.set_cursor_visible(false);
            let _ = grab_platform_cursor(window);
            #[cfg(windows)]
            if !clip_cursor_to_client(window) && !self.clip_warned {
                self.clip_warned = true;
                eprintln!("simu3d: ClipCursor failed; the pointer can leave the window");
            }
            self.pull_pointer_inside(window);
        }

        fn pull_pointer_inside(&mut self, window: &Window) {
            if !self.looking || self.settings_open || !window.has_focus() {
                return;
            }
            #[cfg(windows)]
            {
                let _ = clip_cursor_to_client(window);
            }
            if !cursor_needs_recenter(self.cursor_position, window.inner_size()) {
                return;
            }
            if recenter_cursor(window) {
                self.suppress_cursor_samples = self.suppress_cursor_samples.max(1);
            }
        }

        fn nudge_setting(&mut self, steps: f32) {
            if self.setting_index == KNOBS.len() {
                if steps > 0.0 {
                    self.reload_house();
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

        fn reload_house(&mut self) {
            let Some(sim) = &mut self.sim else {
                return;
            };
            sim.reload_geometry();
            println!(
                "reloaded house at box {:.4} m, {} blocks",
                sim.config.block_size,
                sim.block_count()
            );
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
        Knob { label: "FOV", min: 40.0, max: 110.0, step: 1.0, get: |c| c.fov_degrees, set: |c, v| c.fov_degrees = v },
        Knob { label: "INVERT", min: 0.0, max: 1.0, step: 1.0, get: |c| c.invert_look, set: |c, v| c.invert_look = if v >= 0.5 { 1.0 } else { 0.0 } },
        Knob { label: "GRAVITY", min: 0.5, max: 30.0, step: 0.1, get: |c| c.gravity, set: |c, v| c.gravity = v },
        Knob { label: "LIFETIME", min: 0.5, max: 30.0, step: 0.5, get: |c| c.fragment_lifetime_secs, set: |c, v| c.fragment_lifetime_secs = v },
        Knob { label: "RADIUS", min: 0.15, max: 2.0, step: 0.05, get: |c| c.explosion_radius, set: |c, v| c.explosion_radius = v },
        Knob { label: "IMPULSE", min: 200.0, max: 20000.0, step: 100.0, get: |c| c.blast_impulse, set: |c, v| c.blast_impulse = v },
        Knob { label: "STONE ADH", min: 20.0, max: 400.0, step: 5.0, get: |c| c.adhesive_stone, set: |c, v| c.adhesive_stone = v },
        Knob { label: "GLASS ADH", min: 0.0, max: 40.0, step: 1.0, get: |c| c.adhesive_glass, set: |c, v| c.adhesive_glass = v },
        Knob { label: "ROCKET", min: 10.0, max: 200.0, step: 1.0, get: |c| c.rocket_speed, set: |c, v| c.rocket_speed = v },
        Knob { label: "RELOAD", min: 0.2, max: 5.0, step: 0.05, get: |c| c.rocket_reload_secs, set: |c, v| c.rocket_reload_secs = v },
        Knob { label: "RESERVE", min: 0.0, max: 30.0, step: 1.0, get: |c| c.rocket_reserve as f32, set: |c, v| c.rocket_reserve = v.round().max(0.0) as u32 },
        Knob { label: "SENS", min: SENS_MIN, max: SENS_MAX, step: 0.0002, get: |c| c.mouse_sensitivity, set: |c, v| c.mouse_sensitivity = v },
        Knob { label: "WALK", min: 1.0, max: 12.0, step: 0.1, get: |c| c.walk_speed, set: |c, v| c.walk_speed = v },
        Knob { label: "SPRINT", min: 2.0, max: 20.0, step: 0.1, get: |c| c.sprint_speed, set: |c, v| c.sprint_speed = v },
        Knob { label: "JUMP", min: 1.0, max: 12.0, step: 0.1, get: |c| c.jump_speed, set: |c, v| c.jump_speed = v },
        Knob { label: "HEIGHT", min: 0.8, max: 3.0, step: 0.05, get: |c| c.player_height_m, set: |c, v| c.player_height_m = v },
        Knob { label: "DT", min: 1.0 / 240.0, max: 1.0 / 30.0, step: 0.001, get: |c| c.fixed_dt, set: |c, v| c.fixed_dt = v },
        Knob { label: "BOX", min: 0.1, max: 0.5, step: 0.05, get: |c| c.block_size, set: |c, v| c.block_size = v },
        Knob { label: "JOINT", min: 0.0, max: 0.16, step: 0.004, get: |c| c.block_gap, set: |c, v| c.block_gap = v },
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
            text: "RELOAD GEOMETRY".to_string(),
            selected: selected == KNOBS.len(),
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
    ) -> Option<(f64, f64)> {
        if *suppress > 0 {
            *suppress -= 1;
            *previous = Some(current);
            return None;
        }
        let delta = previous.map(|prev| (current.0 - prev.0, current.1 - prev.1));
        *previous = Some(current);
        delta
    }

    /// Raw device motion is the look source once any non-zero sample has arrived.
    /// Cursor-position deltas are the fallback. They are not mixed: a warp back
    /// to the window centre would otherwise cancel the raw delta, or be chosen
    /// on the axis the warp just moved.
    fn select_look_counts(raw: (f64, f64), cursor: (f64, f64), saw_raw: bool) -> (f64, f64) {
        if saw_raw { raw } else { cursor }
    }

    fn scale_look(dx: f64, dy: f64, sensitivity: f32, invert_look: f32) -> (f32, f32) {
        let pitch_sign = if invert_look >= 0.5 { -1.0 } else { 1.0 };
        (dx as f32 * sensitivity, dy as f32 * sensitivity * pitch_sign)
    }

    fn cursor_needs_recenter(position: Option<(f64, f64)>, size: PhysicalSize<u32>) -> bool {
        let Some((x, y)) = position else {
            return true;
        };
        if size.width < 2 || size.height < 2 {
            return false;
        }
        let cx = size.width as f64 * 0.5;
        let cy = size.height as f64 * 0.5;
        (x - cx).abs() > 1.0 || (y - cy).abs() > 1.0
    }

    fn recenter_cursor(window: &Window) -> bool {
        let size = window.inner_size();
        if size.width < 2 || size.height < 2 {
            return false;
        }
        let center = PhysicalPosition::new(size.width as f64 * 0.5, size.height as f64 * 0.5);
        window.set_cursor_position(center).is_ok()
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
            window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
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
                let mut client = Rect { left: 0, top: 0, right: 0, bottom: 0 };
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
                let mut current = Rect { left: 0, top: 0, right: 0, bottom: 0 };
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
            assert_eq!(take_cursor_delta(&mut suppress, &mut previous, (800.0, 450.0)), None);
            assert_eq!(suppress, 0);
            assert_eq!(
                take_cursor_delta(&mut suppress, &mut previous, (820.0, 440.0)),
                Some((20.0, -10.0))
            );
        }

        #[test]
        fn raw_motion_is_not_mixed_with_the_recenter_warp() {
            let raw = (12.0, -4.0);
            let warp = (-12.0, 4.0);
            assert_eq!(select_look_counts(raw, warp, true), raw);
            assert_eq!(select_look_counts((0.0, 0.0), warp, false), warp);
        }

        #[test]
        fn sensitivity_panel_reaches_past_the_old_ceiling() {
            assert!(SENS_MAX > 0.01);
            assert!(SENS_MIN < 0.0004);
            let up = step_knob("SENS", 0.0022, 1.0, 0.0002, SENS_MIN, SENS_MAX);
            assert!((up - 0.0033).abs() < 1.0e-6);
            assert_eq!(step_knob("SENS", 0.04, 1.0, 0.0002, SENS_MIN, SENS_MAX), SENS_MAX);
            assert_eq!(step_knob("SENS", 0.00012, -1.0, 0.0002, SENS_MIN, SENS_MAX), SENS_MIN);
            assert_eq!(step_knob("WALK", 4.3, 1.0, 0.1, 1.0, 12.0), 4.4);
        }

        #[test]
        fn recenter_when_the_pointer_is_away_from_the_middle() {
            let size = PhysicalSize::new(1600, 900);
            assert!(cursor_needs_recenter(None, size));
            assert!(!cursor_needs_recenter(Some((800.0, 450.0)), size));
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
