mod game;
mod main_menu;
mod systems;

use bevy::{app::App, prelude::*};
use main_menu::MainMenuPlugin;
use systems::transition_game_and_main_menu;

use crate::game::GamePlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_state::<AppState>()
        .add_plugin(GamePlugin)
        .add_plugin(MainMenuPlugin)
        .add_system(transition_game_and_main_menu)
        .run();
}

#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AppState {
    #[default]
    MainMenu,
    Game,
    GameOver,
}
