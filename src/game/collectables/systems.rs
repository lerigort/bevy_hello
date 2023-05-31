use bevy::{prelude::*, transform::commands, window::PrimaryWindow};
use rand::prelude::*;
use std::time::Duration;

use crate::{
    game::{collectables::events::SetConcreteModeActive, SimulationState},
    AppState,
};

use super::{
    components::*,
    events::*,
    resources::*,
    *,
};

fn initial_spawn_collectable(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    window_query: Query<&Window, With<PrimaryWindow>>,
) {
    let window = window_query.get_single().unwrap();
    for _ in 0..INITIAL_AMOUNT_OF_COLLECTABLES {
        commands.spawn((
            SpriteBundle {
                texture: asset_server.load("kenney_pixel-platformer/Characters/character_0013.png"),
                transform: Transform::from_xyz(
                    random::<f32>() * window.width(),
                    random::<f32>() * window.height(),
                    0.0,
                ),
                ..default()
            },
            Collectable,
        ));
    }
}

// creates one separate collectable -- intended to use with events
fn spawn_collectable_by_event(
    mut commands: Commands,
    window_query: Query<&Window, With<PrimaryWindow>>,
    asset_server: Res<AssetServer>,
    mut event_reader: EventReader<AddCollectableEvent>,
) {
    let window = window_query.get_single().unwrap();
    // we can have few EVENTS in one frame, proceed throught all of them
    for event in event_reader.iter() {
        // and every event hold value -- how much collectables to spawn
        for _ in 0..event.amount {
            commands.spawn((
                SpriteBundle {
                    texture: asset_server
                        .load("kenney_pixel-platformer/Characters/character_0013.png"),
                    transform: Transform::from_xyz(
                        random::<f32>() * window.width(),
                        random::<f32>() * window.height(),
                        0.0,
                    ),
                    ..default()
                },
                Collectable,
            ));
        }
    }
}

pub fn check_collision_with_collectable(
    mut score_holder: ResMut<Score>,
    mut commands: Commands,
    query_concrete_entity: Query<Entity, With<ConcreteModeCollectable>>,
    query_collectable: Query<(&Transform, Entity), With<Collectable>>,
    query_player: Query<&Transform, With<Player>>,
    asset_server: Res<AssetServer>,
    audio: Res<Audio>,
    mut eventwriter: EventWriter<SetConcreteModeActive>,
) {
    if let Ok(player_position) = query_player.get_single() {
        let mut all_concrete_entities = Vec::new();
        for entity in query_concrete_entity.iter() {
            all_concrete_entities.push(entity);
        }

        for (collectable_position, collectable) in query_collectable.iter() {
            let distance = collectable_position
                .translation
                .distance(player_position.translation);
            if distance <= (COLLECTABLE_SIZE / 2.0 + PLAYER_SIZE / 2.0) {
                commands.entity(collectable).despawn();
                let sound_collected = asset_server.load("sound_effects/Fruit collect 1_01.ogg");
                audio.play(sound_collected);
                score_holder.score += 1;
                println!("Score is: {}", score_holder.score);

                // if collectable IS Concrete_Mode collectable
                if all_concrete_entities.contains(&collectable) {
                    // send ivent to system "go_into_concrete_mode"
                    eventwriter.send(SetConcreteModeActive {});
                    println!("YOU ARE IN CONCRETE MODE!!!");
                }
            }
        }
    }
}

fn go_into_concrete_mode(
    mut event_reader: EventReader<SetConcreteModeActive>,
    mut commands: Commands,
    mut concrete_mode_holder: ResMut<IsInConcreteMode>,
    player_query: Query<(Entity, &Transform), With<Player>>,
    asset_server: Res<AssetServer>,
) {
    //recive event and consume it
    for _ in event_reader.iter() {
        // look at the Player => systems => "collision_check"
        // this is where the logic of concrete_mode implemented
        concrete_mode_holder.status = true;
        // start timer
        spawn_timer_for_concrete_mode(&mut commands);


        // change skin of player, while in concrete mode
        if let Ok((player_entity, player_position)) = player_query.get_single() {
            commands.entity(player_entity).insert(
                SpriteBundle {
                    transform: *player_position,
                    texture: asset_server.load("kenney_pixel-platformer/Characters/character_0004.png"),
                    ..default()
                },
            );
        }
    }
}

fn spawn_timer_for_concrete_mode(commands: &mut Commands) {
    commands.spawn(ConcreteModeTimer {
        timer: Timer::new(Duration::from_secs(CONCRETE_MODE_DURATION), TimerMode::Once),
        almost_elapsed: false
    });
}

fn spawn_timer_for_elapsing_concrete_mode(commands: &mut Commands) {
    commands.spawn(ConcreteModeElapsingTimer {
        timer: Timer::new(Duration::from_millis(200), TimerMode::Repeating),
        player_skin: PlayerSkin::Concrete,
    });
}

// ticks and handle timer of concrete mode 
fn tick_timer_for_concrete_mode(
    mut query_concrete_timer: Query<(Entity, &mut ConcreteModeTimer)>,
    mut concrete_mode_holder: ResMut<IsInConcreteMode>,
    time: Res<Time>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    player_query: Query<(Entity, &Transform), With<Player>>,
) {
    for (concrete_timer_entity, mut concrete_timer) in query_concrete_timer.iter_mut()  {
        concrete_timer.timer.tick(time.delta());

        if (concrete_timer.timer.percent_left() < 0.2) & !(concrete_timer.almost_elapsed) {
            // we need to do it once, so there is come in handy flag "almost elapsed"
            spawn_timer_for_elapsing_concrete_mode(&mut commands);
            concrete_timer.almost_elapsed = true;
        }

        if concrete_timer.timer.just_finished() {
            concrete_mode_holder.status = false;
            println!("You out of concrete mode!");
            // despawn concrete_mode timer
            commands.entity(concrete_timer_entity).despawn();

            // change skin of player, while out of concrete mode
            if let Ok((player_entity, player_position)) = player_query.get_single() {
                commands.entity(player_entity).insert(
                    SpriteBundle {
                        transform: *player_position,
                        texture: asset_server.load("kenney_pixel-platformer/Characters/character_0000_l.png"),
                        ..default()
                    },
                );
            }
        }   
    }

}

// this timer helps to create animation of elapsing concrete mode
fn tick_timer_for_elapsing_concrete_mode(
    mut query_concrete_elapsing_timer: Query<(&mut ConcreteModeElapsingTimer, Entity)>,
    time: Res<Time>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    player_query: Query<(Entity, &Transform), With<Player>>,
    is_concrete_mode_on: Res<IsInConcreteMode>

) {
    // TODO: there is possibility of multiple concrete modes - but! I dont want interference between animations
    if let Ok((mut concrete_elapsing_timer, concrete_elapsing_timer_entity)) = query_concrete_elapsing_timer.get_single_mut() {
        
        // if player out of concrete mode - we dont need animation no more, so we need to despawn the timer
        // also player skin is alredy default (because of other system) - therefore no additional actions are needed
        if is_concrete_mode_on.status == false {
            commands.entity(concrete_elapsing_timer_entity).despawn();
            return ();
        }
        
        concrete_elapsing_timer.timer.tick(time.delta());

        // timer is repetitive, every time it finishes - change the skin of player
        if concrete_elapsing_timer.timer.finished() {
            if let Ok((player_entity, player_position)) = player_query.get_single() {
                
                if concrete_elapsing_timer.player_skin == PlayerSkin::Concrete {
                    concrete_elapsing_timer.player_skin = PlayerSkin::Usual; 
                    commands.entity(player_entity).insert(
                        SpriteBundle {
                            transform: *player_position,
                            texture: asset_server.load("kenney_pixel-platformer/Characters/character_0000_l.png"),
                            ..default()
                        },
                    );
                }
                else if concrete_elapsing_timer.player_skin == PlayerSkin::Usual {
                    concrete_elapsing_timer.player_skin = PlayerSkin::Concrete; 
                    commands.entity(player_entity).insert(
                        SpriteBundle {
                            transform: *player_position,
                            texture: asset_server.load("kenney_pixel-platformer/Characters/character_0004.png"),
                            ..default()
                        },
                    );
                }                
            }            
        }
    }

}

//every collected point respawn 1 guaranteed collectable, and 2 - with 50% chance.
fn respawn_collectable_by_increasing_score(
    score_holder: Res<Score>,
    mut event_writer: EventWriter<AddCollectableEvent>,
) {
    if score_holder.is_changed() {
        let add_two_collectables = random::<bool>();
        if add_two_collectables {
            event_writer.send(AddCollectableEvent { amount: (2) })
        } else {
            event_writer.send(AddCollectableEvent { amount: (1) })
        }
        // important note -- every event lives until end of next frame or until it has been readen
        // so we dont need clear it explicitly, also code prevents accumulating of values
    }
}

fn increase_chance_of_concrete_mode_by_increasing_score(
    mut chance_of_concrete_mode: ResMut<ConcreteChance>,
    score_holder: Res<Score>,
) {
    if score_holder.is_changed() {
        chance_of_concrete_mode.chance += RISE_STEP_OF_CHANCE_OF_CONCRETE_MODE;
    }
}

fn try_spawn_concrete_mode_collectable(
    window_query: Query<&Window, With<PrimaryWindow>>,
    asset_server: Res<AssetServer>,
    mut chance_of_concrete_mode: ResMut<ConcreteChance>,
    score_holder: Res<Score>,
    mut commands: Commands,
) {
    if score_holder.is_changed() {
        let chance_to_random_spawn = random::<f32>();
        let window = window_query.get_single().unwrap();

        if chance_to_random_spawn <= chance_of_concrete_mode.chance {
            commands.spawn((
                SpriteBundle {
                    texture: asset_server
                        .load("kenney_pixel-platformer/Characters/character_0008.png"),
                    transform: Transform::from_xyz(
                        random::<f32>() * window.width(),
                        random::<f32>() * window.height(),
                        0.0,
                    ),
                    ..default()
                },
                Collectable,
                ConcreteModeCollectable,
            ));
            chance_of_concrete_mode.chance = INITIAL_CHANCE_OF_CONCRETE_MODE;
        }
    }
}

fn create_collectable_holders(mut commands: Commands) {
    commands.insert_resource(Score { score: 0 });
    commands.insert_resource(ConcreteChance {
        chance: INITIAL_CHANCE_OF_CONCRETE_MODE,
    });
    commands.insert_resource(IsInConcreteMode { status: false });
}

fn delete_collectable_holders(mut commands: Commands) {
    commands.remove_resource::<Score>();
    commands.remove_resource::<ConcreteChance>();
    commands.remove_resource::<IsInConcreteMode>();
}

fn despawn_collectables(
    mut commands: Commands,
    collectable_query: Query<Entity, With<Collectable>>,
) {
    for collectable in collectable_query.iter() {
        commands.entity(collectable).despawn();
    }
}

pub struct CollectablesPlugin;

impl Plugin for CollectablesPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            (create_collectable_holders, initial_spawn_collectable)
                .in_schedule(OnEnter(AppState::Game)),
        )
        .add_event::<AddCollectableEvent>()
        .add_event::<SetConcreteModeActive>()
        .add_systems(
            (
                tick_timer_for_elapsing_concrete_mode,
                tick_timer_for_concrete_mode,
                go_into_concrete_mode,
                increase_chance_of_concrete_mode_by_increasing_score,
                check_collision_with_collectable,
                respawn_collectable_by_increasing_score,
                spawn_collectable_by_event,
                try_spawn_concrete_mode_collectable,
            )
                .in_set(OnUpdate(AppState::Game))
                .in_set(OnUpdate(SimulationState::Running)),
                
        )
        .add_systems(
            (delete_collectable_holders, despawn_collectables).in_schedule(OnExit(AppState::Game)),
        );
    }
}
