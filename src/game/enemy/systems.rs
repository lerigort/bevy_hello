use bevy::{prelude::*, window::PrimaryWindow};
use rand::prelude::*;
use std::time::Duration;

use crate::{game::SimulationState, AppState};

use super::{components::*, events::*, *};

fn create_timer_to_change_enemy_movement(mut commands: Commands) {
    commands.insert_resource(resources::ChangeDirectionTimer {
        timer: Timer::new(
            Duration::from_secs(ENEMY_CHANGE_DIRECTION_CYCLE),
            TimerMode::Repeating,
        ),
        sound_not_played: true,
    });
}

fn spawn_enemy(
    mut commands: Commands,
    window_query: Query<&Window, With<PrimaryWindow>>,
    asset_server: Res<AssetServer>,
) {
    let window = window_query.get_single().unwrap();

    for _ in 0..NUMBER_OF_ENEMIES {
        let random_x = random::<f32>() * window.width();
        let random_y = random::<f32>() * window.height();

        commands.spawn((
            SpriteBundle {
                transform: Transform::from_xyz(random_x, random_y, 0.0),
                texture: asset_server.load("kenney_pixel-platformer/Characters/character_0020.png"),
                ..default()
            },
            Enemy {
                direction: Vec2::new(random::<f32>(), random::<f32>()).normalize(),
            },
        ));
    }
}

fn spawn_boss(
    mut commands: Commands,
    window_query: Query<&Window, With<PrimaryWindow>>,
    asset_server: Res<AssetServer>,
    mut event_reader: EventReader<AddBossEvent>,
) {
    let window = window_query.get_single().unwrap();
    let random_x = random::<f32>() * window.width();
    let random_y = random::<f32>() * window.height();

    for _ in event_reader.iter() {
        commands.spawn((
            SpriteBundle {
                transform: Transform::from_xyz(random_x, random_y, 0.0),
                texture: asset_server.load("kenney_pixel-platformer/Characters/character_0015.png"),
                ..default()
            },
            Enemy {
                direction: BOSS_DIRECTION,
            },
            Boss { speed_modifer: 1.0 },
        ));
    }
}

fn boss_movement(mut boss_query: Query<(&mut Transform, &Boss)>, time: Res<Time>) {
    // "Boss" component is speed modificator holder
    for (mut boss_position, boss) in &mut boss_query {
        let mut translation = boss_position.translation;
        let direction = Vec3::new(BOSS_DIRECTION.x, BOSS_DIRECTION.y, 0.0);

        translation += direction * boss.speed_modifer * BOSS_SPEED * time.delta_seconds();

        boss_position.translation = translation;
    }
}

fn enemy_movement(
    mut enemy_query: Query<(&mut Transform, &Enemy), Without<Boss>>,
    time: Res<Time>,
) {
    // "Enemy" component basically is a direction holder
    for (mut transform, enemy) in &mut enemy_query {
        let mut translation = transform.translation;

        let direction = Vec3::new(enemy.direction.x, enemy.direction.y, 0.0);

        translation += direction * ENEMY_SPEED * time.delta_seconds();

        transform.translation = translation;
    }
}

fn check_borders_enemy_movement(
    window_query: Query<&Window, With<PrimaryWindow>>,
    mut enemy_query: Query<(&mut Transform, &Enemy)>,
) {
    let window = window_query.get_single().unwrap();
    for (mut transform, _enemy) in &mut enemy_query {
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

fn spawn_enemies_with_increasing_score(
    mut event_writer: EventWriter<AddEnemyEvent>,
    score: Res<Score>,
) {
    //if we reach 10/20/30.. -> ONCE increase amount of enemies
    if (score.score % 10 == 0) & score.is_changed() {
        event_writer.send(AddEnemyEvent { amount: (3) })
    }
}

fn spawn_boss_with_increasing_score(
    mut event_writer: EventWriter<AddBossEvent>,
    score: Res<Score>,
) {
    //if we reach 30.. -> ONCE add boss
    if (score.score % 30 == 0) & score.is_changed() {
        event_writer.send(AddBossEvent {})
    }
}

fn spawn_enemies_throught_event(
    mut commands: Commands,
    window_query: Query<&Window, With<PrimaryWindow>>,
    asset_server: Res<AssetServer>,
    mut event_reader: EventReader<AddEnemyEvent>,
) {
    let window = window_query.get_single().unwrap();
    // we can have few EVENTS in one frame, proceed throught all of them
    for event in event_reader.iter() {
        // and every event hold value -- how much enemies to spawn
        for _ in 0..event.amount {
            let random_x = random::<f32>() * window.width();
            let random_y = random::<f32>() * window.height();

            commands.spawn((
                SpriteBundle {
                    transform: Transform::from_xyz(random_x, random_y, 0.0),
                    texture: asset_server
                        .load("kenney_pixel-platformer/Characters/character_0020.png"),
                    ..default()
                },
                Enemy {
                    direction: Vec2::new(random::<f32>(), random::<f32>()).normalize(),
                },
            ));
        }
    }
}

fn change_enemy_movement_with_timer(
    mut enemy_query: Query<&mut Enemy, Without<Boss>>,
    time: Res<Time>,
    mut config: ResMut<ChangeDirectionTimer>,
    audio: Res<Audio>,
    asset_server: Res<AssetServer>,
    mut event_writer: EventWriter<AddEnemyEvent>,
    mut event_writer_boss_acceleration: EventWriter<SpeedUpBoss>,
) {
    config.timer.tick(time.delta());

    if (config.timer.percent_left() < 0.1) & config.sound_not_played & !(enemy_query.is_empty()) {
        audio.play(asset_server.load("sound_effects/Drum_01.ogg"));
        config.sound_not_played = false;
    }

    if config.timer.finished() {
        config.sound_not_played = true;
        for mut enemy in &mut enemy_query {
            enemy.direction = Vec2::new(random::<f32>(), random::<f32>()).normalize();

            let is_x_negative = random::<bool>();
            let is_y_negative = random::<bool>();
            if is_x_negative {
                enemy.direction.x = -(enemy.direction.x);
            }
            if is_y_negative {
                enemy.direction.y = -(enemy.direction.y);
            }
        }
        // Every 5 seconds add 1 usual enemy
        // It is not the only way to increasing amount of enemies
        event_writer.send(AddEnemyEvent { amount: (1) });

        // Also, to this timer we bind speed up for boss
        // Timer tick and timer handle happens inside this system,
        // so there is no good way to separate functionality
        event_writer_boss_acceleration.send(SpeedUpBoss {});
    }
}

fn handle_speed_up_boss(
    mut boss_query: Query<&mut Boss>,
    mut event_reader: EventReader<SpeedUpBoss>,
) {
    for _ in event_reader.iter() {
        for mut boss in boss_query.iter_mut() {
            boss.speed_modifer += 0.1;
        }
    }
}

fn despawn_enemy(mut commands: Commands, enemy_query: Query<Entity, With<Enemy>>) {
    for enemy in enemy_query.iter() {
        commands.entity(enemy).despawn();
    }
}

pub struct EnemyPlugin;

impl Plugin for EnemyPlugin {
    fn build(&self, app: &mut App) {
        app.add_startup_system(create_timer_to_change_enemy_movement)
            .add_system(spawn_enemy.in_schedule(OnEnter(AppState::Game)))
            .add_event::<AddEnemyEvent>()
            .add_event::<AddBossEvent>()
            .add_event::<SpeedUpBoss>()
            .add_systems(
                (
                    boss_movement,
                    enemy_movement,
                    check_borders_enemy_movement,
                    change_enemy_movement_with_timer,
                    handle_speed_up_boss,
                    spawn_enemies_throught_event,
                    spawn_enemies_with_increasing_score,
                    spawn_boss_with_increasing_score,
                    spawn_boss,
                )
                    .in_set(OnUpdate(AppState::Game))
                    .in_set(OnUpdate(SimulationState::Running)),
            )
            .add_system(despawn_enemy.in_schedule(OnExit(AppState::Game)));
    }
}
