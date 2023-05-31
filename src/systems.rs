use bevy::prelude::*;

use crate::AppState;

pub fn transition_game_and_main_menu(
    mut commands: Commands,
    keyboard_input: Res<Input<KeyCode>>,
    app_state: Res<State<AppState>>,
) {
    if keyboard_input.just_pressed(KeyCode::G) & (app_state.0 != AppState::Game) {
        commands.insert_resource(NextState(Some(AppState::Game)));
        println!("Entered GAME state");
    } else if keyboard_input.just_pressed(KeyCode::M) & (app_state.0 != AppState::MainMenu) {
        commands.insert_resource(NextState(Some(AppState::MainMenu)));
        println!("Entered MAIN MENU state");
    }
}
