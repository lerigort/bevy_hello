use bevy::{prelude::*, window::PrimaryWindow};

use crate::{game::enemy::components::Enemy, AppState};

use super::events::GameOver;

fn handle_game_over(
    mut game_over_event: EventReader<GameOver>,
    enemy_query: Query<Entity, With<Enemy>>,
    mut commands: Commands,
) {
    for _event in game_over_event.iter() {
        for enemy in enemy_query.iter() {
            commands.entity(enemy).despawn();
        }
        commands.insert_resource(NextState(Some(AppState::GameOver)));
        break;
    }
}

fn spawn_camera(mut commands: Commands, window_query: Query<&Window, With<PrimaryWindow>>) {
    let window = window_query.get_single().unwrap();
    commands.spawn(Camera2dBundle {
        transform: Transform::from_xyz(window.width() / 2.0, window.height() / 2.0, 15.0),
        ..default()
    });
}

pub struct MechanicsPlugin;

impl Plugin for MechanicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_startup_system(spawn_camera)
            .add_event::<GameOver>()
            .add_system(handle_game_over);
    }
}
