//! Runtime knobs for gravity, fragment size and count, and blast behavior.

use std::path::Path;

#[derive(Clone, Debug)]
pub struct Config {
    pub gravity: f32,
    pub fragment_count: usize,
    pub fragment_lifetime_secs: f32,
    pub min_fragment_extent: f32,
    pub explosion_radius: f32,
    pub blast_impulse: f32,
    pub rocket_speed: f32,
    pub rocket_reload_secs: f32,
    pub rocket_reserve: u32,
    pub block_size: f32,
    pub fixed_dt: f32,
    /// Radians per mouse count. The settings panel covers 0.0001 to 0.05.
    pub mouse_sensitivity: f32,
    pub walk_speed: f32,
    pub sprint_speed: f32,
    pub jump_speed: f32,
    pub player_height_m: f32,
    pub fov_degrees: f32,
    /// 1 = right-mouse look is inverted on the vertical axis. 0 = mouse down looks down.
    pub invert_look: f32,
    pub adhesive_stone: f32,
    pub adhesive_glass: f32,
    pub adhesive_concrete: f32,
    pub adhesive_plaster: f32,
    pub adhesive_wood: f32,
    pub adhesive_wood_dark: f32,
    pub adhesive_metal: f32,
    pub adhesive_grass: f32,
    pub adhesive_soil: f32,
    pub adhesive_roof: f32,
    pub adhesive_tile: f32,
    pub adhesive_fabric: f32,
    pub adhesive_linen: f32,
    pub adhesive_water: f32,
    /// Full gap between two block faces, as a fraction of the block edge.
    pub block_gap: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            gravity: 9.80665,
            fragment_count: 1000,
            fragment_lifetime_secs: 5.0,
            min_fragment_extent: 0.005,
            explosion_radius: 1.5,
            blast_impulse: 10000.0,
            rocket_speed: 80.0,
            rocket_reload_secs: 0.3,
            rocket_reserve: 30,
            // 10 × cbrt(10 cm³). Each of the 1000 children is a 10 cm³ cube.
            block_size: 10.0 * 1.0e-5_f32.cbrt(),
            fixed_dt: 1.0 / 120.0,
            mouse_sensitivity: 0.0022,
            walk_speed: 4.3,
            sprint_speed: 7.2,
            jump_speed: 4.7,
            player_height_m: 1.7,
            fov_degrees: 110.0,
            invert_look: 1.0,
            // N·s a face-bond holds. The panel default is a weak stone bond and no glass bond.
            adhesive_stone: 20.0,
            adhesive_glass: 0.0,
            adhesive_concrete: 55.0,
            adhesive_plaster: 28.0,
            adhesive_wood: 22.0,
            adhesive_wood_dark: 26.0,
            adhesive_metal: 260.0,
            adhesive_grass: 8.0,
            adhesive_soil: 14.0,
            adhesive_roof: 36.0,
            adhesive_tile: 48.0,
            adhesive_fabric: 4.0,
            adhesive_linen: 4.0,
            adhesive_water: 0.0,
            block_gap: 0.004,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunMode {
    Window,
    SelfTest,
    Stress,
}

use crate::material::Material;

impl Config {
    /// Impulse, N·s, that one face-bond of this material holds.
    pub fn adhesive(&self, material: Material) -> f32 {
        match material {
            Material::Stone => self.adhesive_stone,
            Material::Glass => self.adhesive_glass,
            Material::Concrete => self.adhesive_concrete,
            Material::Plaster => self.adhesive_plaster,
            Material::Wood => self.adhesive_wood,
            Material::WoodDark => self.adhesive_wood_dark,
            Material::Metal => self.adhesive_metal,
            Material::Grass => self.adhesive_grass,
            Material::Soil => self.adhesive_soil,
            Material::Roof => self.adhesive_roof,
            Material::Tile => self.adhesive_tile,
            Material::Fabric => self.adhesive_fabric,
            Material::Linen => self.adhesive_linen,
            Material::Water => self.adhesive_water,
        }
    }

    pub fn cube_half(&self, full_edge: f32) -> f32 {
        full_edge * (1.0 - self.block_gap) * 0.5
    }

    pub fn cube_bevel(&self) -> f32 {
        (self.block_gap * 1.5).clamp(0.0, 0.4)
    }

    pub fn from_args() -> (Self, RunMode) {
        let mut cfg = Self::load_default_file();
        let mut mode = RunMode::Window;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--self-test" => mode = RunMode::SelfTest,
                "--stress" => mode = RunMode::Stress,
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                "--config" => {
                    let path = args.next().unwrap_or_else(|| missing("--config"));
                    cfg = Self::load_file(Path::new(&path)).unwrap_or_else(|err| {
                        eprintln!("simu3d: {err}");
                        std::process::exit(2);
                    });
                }
                "--gravity" => cfg.gravity = number(&mut args, "--gravity"),
                "--fragment-count" => {
                    cfg.fragment_count = number::<f32>(&mut args, "--fragment-count") as usize
                }
                "--fragment-lifetime" => {
                    cfg.fragment_lifetime_secs = number(&mut args, "--fragment-lifetime")
                }
                "--min-fragment-extent" => {
                    cfg.min_fragment_extent = number(&mut args, "--min-fragment-extent")
                }
                "--explosion-radius" => {
                    cfg.explosion_radius = number(&mut args, "--explosion-radius")
                }
                "--blast-impulse" => cfg.blast_impulse = number(&mut args, "--blast-impulse"),
                "--rocket-speed" => cfg.rocket_speed = number(&mut args, "--rocket-speed"),
                "--reload" => cfg.rocket_reload_secs = number(&mut args, "--reload"),
                "--mouse-sensitivity" => {
                    cfg.mouse_sensitivity = number(&mut args, "--mouse-sensitivity")
                }
                "--block-size" | "--box-size" => cfg.block_size = number(&mut args, "--box-size"),
                "--player-height" => cfg.player_height_m = number(&mut args, "--player-height"),
                other => {
                    eprintln!("simu3d: unknown argument {other}");
                    print_help();
                    std::process::exit(2);
                }
            }
        }
        if cfg.fragment_count == 0 {
            eprintln!("simu3d: fragment_count must be at least 1");
            std::process::exit(2);
        }
        if cfg.gravity <= 0.0 || cfg.fixed_dt <= 0.0 || cfg.block_size <= 0.0 {
            eprintln!("simu3d: gravity, fixed_dt, and block_size must be positive");
            std::process::exit(2);
        }
        if !(0.8..=3.0).contains(&cfg.player_height_m) {
            eprintln!("simu3d: player height must be between 0.8 and 3.0 metres");
            std::process::exit(2);
        }
        (cfg, mode)
    }

    fn load_default_file() -> Self {
        let cwd = Path::new("simu3d.toml");
        if cwd.is_file() {
            match Self::load_file(cwd) {
                Ok(cfg) => return cfg,
                Err(err) => eprintln!("simu3d: {err} — using defaults"),
            }
        }
        Self::default()
    }

    pub fn load_file(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|err| format!("cannot read {}: {err}", path.display()))?;
        let mut cfg = Self::default();
        for (line_no, raw) in text.lines().enumerate() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                return Err(format!("{}:{}: expected key = value", path.display(), line_no + 1));
            };
            let key = key.trim();
            let value = value.trim();
            let num: f32 = value.parse().map_err(|_| {
                format!("{}:{}: {key} is not a number", path.display(), line_no + 1)
            })?;
            match key {
                "gravity" => cfg.gravity = num,
                "fragment_count" => cfg.fragment_count = num as usize,
                "fragment_lifetime_secs" => cfg.fragment_lifetime_secs = num,
                "min_fragment_extent" => cfg.min_fragment_extent = num,
                "explosion_radius" => cfg.explosion_radius = num,
                "blast_impulse" => cfg.blast_impulse = num,
                "rocket_speed" => cfg.rocket_speed = num,
                "rocket_reload_secs" => cfg.rocket_reload_secs = num,
                "rocket_reserve" => cfg.rocket_reserve = num as u32,
                "block_size" | "box_size" => cfg.block_size = num,
                "fixed_dt" => cfg.fixed_dt = num,
                "mouse_sensitivity" => cfg.mouse_sensitivity = num,
                "walk_speed" => cfg.walk_speed = num,
                "sprint_speed" => cfg.sprint_speed = num,
                "jump_speed" => cfg.jump_speed = num,
                "player_height_m" => cfg.player_height_m = num,
                "fov_degrees" => cfg.fov_degrees = num,
                "invert_look" => cfg.invert_look = num,
                "adhesive_stone" => cfg.adhesive_stone = num,
                "adhesive_glass" => cfg.adhesive_glass = num,
                "adhesive_concrete" => cfg.adhesive_concrete = num,
                "adhesive_plaster" => cfg.adhesive_plaster = num,
                "adhesive_wood" => cfg.adhesive_wood = num,
                "adhesive_wood_dark" => cfg.adhesive_wood_dark = num,
                "adhesive_metal" => cfg.adhesive_metal = num,
                "adhesive_grass" => cfg.adhesive_grass = num,
                "adhesive_soil" => cfg.adhesive_soil = num,
                "adhesive_roof" => cfg.adhesive_roof = num,
                "adhesive_tile" => cfg.adhesive_tile = num,
                "adhesive_fabric" => cfg.adhesive_fabric = num,
                "adhesive_linen" => cfg.adhesive_linen = num,
                "adhesive_water" => cfg.adhesive_water = num,
                "block_gap" => cfg.block_gap = num,
                other => {
                    return Err(format!("{}:{}: unknown key {other}", path.display(), line_no + 1));
                }
            }
        }
        Ok(cfg)
    }

    pub fn save_default(&self) -> std::io::Result<()> {
        let text = format!(
            "# Written by the in-game settings panel. Restart is not required except for block_size.\n\
             gravity = {gravity}\n\
             fragment_count = {fragment_count}\n\
             fragment_lifetime_secs = {lifetime}\n\
             min_fragment_extent = {min_extent}\n\
             explosion_radius = {radius}\n\
             blast_impulse = {impulse}\n\
             rocket_speed = {rocket_speed}\n\
             rocket_reload_secs = {reload}\n\
             rocket_reserve = {reserve}\n\
             block_size = {block_size}\n\
             box_size = {block_size}\n\
             fixed_dt = {dt}\n\
             mouse_sensitivity = {sens}\n\
             walk_speed = {walk}\n\
             sprint_speed = {sprint}\n\
             jump_speed = {jump}\n\
             player_height_m = {player_height}\n\
             fov_degrees = {fov}\n\
             invert_look = {invert}\n\
             adhesive_stone = {adhesive_stone}\n\
             adhesive_glass = {adhesive_glass}\n\
             adhesive_concrete = {adhesive_concrete}\n\
             adhesive_plaster = {adhesive_plaster}\n\
             adhesive_wood = {adhesive_wood}\n\
             adhesive_wood_dark = {adhesive_wood_dark}\n\
             adhesive_metal = {adhesive_metal}\n\
             adhesive_grass = {adhesive_grass}\n\
             adhesive_soil = {adhesive_soil}\n\
             adhesive_roof = {adhesive_roof}\n\
             adhesive_tile = {adhesive_tile}\n\
             adhesive_fabric = {adhesive_fabric}\n\
             adhesive_linen = {adhesive_linen}\n\
             adhesive_water = {adhesive_water}\n\
             block_gap = {block_gap}\n",
            gravity = self.gravity,
            fragment_count = self.fragment_count,
            lifetime = self.fragment_lifetime_secs,
            min_extent = self.min_fragment_extent,
            radius = self.explosion_radius,
            impulse = self.blast_impulse,
            rocket_speed = self.rocket_speed,
            reload = self.rocket_reload_secs,
            reserve = self.rocket_reserve,
            block_size = self.block_size,
            dt = self.fixed_dt,
            sens = self.mouse_sensitivity,
            walk = self.walk_speed,
            sprint = self.sprint_speed,
            jump = self.jump_speed,
            player_height = self.player_height_m,
            fov = self.fov_degrees,
            invert = self.invert_look,
            adhesive_stone = self.adhesive_stone,
            adhesive_glass = self.adhesive_glass,
            adhesive_concrete = self.adhesive_concrete,
            adhesive_plaster = self.adhesive_plaster,
            adhesive_wood = self.adhesive_wood,
            adhesive_wood_dark = self.adhesive_wood_dark,
            adhesive_metal = self.adhesive_metal,
            adhesive_grass = self.adhesive_grass,
            adhesive_soil = self.adhesive_soil,
            adhesive_roof = self.adhesive_roof,
            adhesive_tile = self.adhesive_tile,
            adhesive_fabric = self.adhesive_fabric,
            adhesive_linen = self.adhesive_linen,
            adhesive_water = self.adhesive_water,
            block_gap = self.block_gap,
        );
        std::fs::write("simu3d.toml", text)
    }
}

fn number<T: std::str::FromStr>(args: &mut impl Iterator<Item = String>, flag: &str) -> T {
    let raw = args.next().unwrap_or_else(|| missing(flag));
    raw.parse().unwrap_or_else(|_| {
        eprintln!("simu3d: {flag} expects a number, got {raw}");
        std::process::exit(2);
    })
}

fn missing(flag: &str) -> String {
    eprintln!("simu3d: {flag} needs a value");
    std::process::exit(2);
}

fn print_help() {
    eprintln!(
        "simu3d — voxel house, gravity, bazooka\n\
         \n\
         cargo run --release\n\
         cargo run --release -- --self-test\n\
         \n\
         --fragment-lifetime <seconds>   default 5\n\
         --box-size <metres>             big box edge. The 10 cm³ cubes are not a setting.\n\
         --gravity <m/s^2>               default 9.80665\n\
         --explosion-radius <metres>\n\
         --player-height <metres>          default 1.7\n\
         --mouse-sensitivity <radians per count>  default 0.0022; panel 0.0001 to 0.05\n\
         --config <file>\n\
         \n\
         ZQSD move. AZERTY letters, not WASD. Mouse look, Space jump, Shift sprint.\n\
         Left mouse fires the bazooka. R reloads. Esc releases the mouse. F3 prints counts."
    );
}
