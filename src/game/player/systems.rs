use bevy::{prelude::*, window::PrimaryWindow};

use super::components::Player;
use super::*;

use crate::game::collectables::resources::IsInConcreteMode;
use crate::game::enemy::ENEMY_SIZE;
use crate::game::SimulationState;
use crate::AppState;

fn spawn_player(
    mut commands: Commands,
    window_query: Query<&Window, With<PrimaryWindow>>,
    asset_server: Res<AssetServer>,
) {
    let window = window_query.get_single().unwrap();
    commands.spawn((
        SpriteBundle {
            transform: Transform::from_xyz(window.width() / 2.0, window.height() / 2.0, 0.0),
            texture: asset_server.load("kenney_pixel-platformer/Characters/character_0000_l.png"),
            ..default()
        },
        Player {},
    ));
}

fn player_movement(
    keyboard_input: Res<Input<KeyCode>>,
    mut player_query: Query<&mut Transform, With<Player>>,
    time: Res<Time>,
) {
    if let Ok(mut transform) = player_query.get_single_mut() {
        let mut direction = Vec3::ZERO;
        if keyboard_input.pressed(KeyCode::W) {
            direction += Vec3::new(0.0, 1.0, 0.0);
        }
        if keyboard_input.pressed(KeyCode::A) {
            direction += Vec3::new(-1.0, 0.0, 0.0);
        }
        if keyboard_input.pressed(KeyCode::S) {
            direction += Vec3::new(0.0, -1.0, 0.0);
        }
        if keyboard_input.pressed(KeyCode::D) {
            direction += Vec3::new(1.0, 0.0, 0.0);
        }

        if direction.length() > 0.0 {
            direction = direction.normalize();
        }

        transform.translation += direction * PLAYER_SPEED * time.delta_seconds();
    }
}

fn check_borders_player_movement(
    window_query: Query<&Window, With<PrimaryWindow>>,
    mut player_query: Query<&mut Transform, With<Player>>,
) {
    if let Ok(mut transform) = player_query.get_single_mut() {
        let window = window_query.get_single().unwrap();
        let x_min = 0.0;
        let x_max = window.width();
        let y_min = 0.0;
        let y_max = window.height();

        let mut translation = transform.translation;

        if translation.x < x_min {
            translation.x = x_max;
        } else if translation.x > x_max {
            translation.x = x_min;
        }

        if translation.y < y_min {
            translation.y = y_max;
        } else if translation.y > y_max {
            translation.y = y_min;
        }

        transform.translation = translation;
    }
}

fn collision_check(
    mut player_single_query: Query<(Entity, &Transform), With<Player>>,
    enemy_query: Query<(Entity, &Transform), With<Enemy>>,
    mut commands: Commands,
    mut game_over_event: EventWriter<GameOver>,
    mut concrete_mode_check: Res<IsInConcreteMode>,
) {
    if let Ok((player, player_position)) = player_single_query.get_single_mut() {
        let player_radius = PLAYER_SIZE / 2.0;
        let enemy_radius = ENEMY_SIZE / 2.0;

        for (enemy, enemy_positison) in enemy_query.iter() {
            let distance = player_position
                .translation
                .distance(enemy_positison.translation);

            if distance <= (player_radius + enemy_radius) {
                // "realisation of concrete_mode"
                match concrete_mode_check.status {
                    false => {
                        println!("Game over!");
                        game_over_event.send(GameOver(true));
                        commands.entity(player).despawn();
                    }
                    true => {
                        commands.entity(enemy).despawn();
                    }
                }
            }
        }
    }
}

fn despawn_player(mut commands: Commands, player_query: Query<Entity, With<Player>>) {
    if let Ok(player) = player_query.get_single() {
        commands.entity(player).despawn();
    }
}

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_system(spawn_player.in_schedule(OnEnter(AppState::Game)))
            .add_systems(
                (
                    collision_check,
                    check_borders_player_movement,
                    player_movement,
                )
                    .in_set(OnUpdate(AppState::Game))
                    .in_set(OnUpdate(SimulationState::Running)),
            )
            .add_system(despawn_player.in_schedule(OnExit(AppState::Game)));
    }
}
