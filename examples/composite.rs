use bevy::{
    color::palettes::css::RED, prelude::*, render::view::Hdr, sprite::Anchor, window::PrimaryWindow,
};
use bevy_firefly::{composite_sprite::CompositeSpriteSlice, data::NormalMode, prelude::*};

fn main() {
    let mut app = App::new();

    app.add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()));
    app.add_plugins((FireflyPlugin /*FireflyGizmosPlugin*/,));

    app.init_resource::<Dragged>();

    app.add_systems(Startup, setup);
    app.add_systems(Update, (z_sorting, drag_objects));

    app.run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let mut proj = OrthographicProjection::default_2d();
    proj.scale = 0.15;

    commands.spawn((
        Camera2d,
        Hdr,
        Projection::Orthographic(proj),
        FireflyConfig {
            // normal maps need to be explicitly enabled
            normal_mode: NormalMode::TopDown,
            ..default()
        },
    ));

    let texture_atlas = TextureAtlasLayout::from_grid(UVec2::splat(18), 1, 2, None, None);
    let atlas_handle = asset_server.add(texture_atlas);

    commands.spawn((
        Sprite::from_atlas_image(
            asset_server.load("composite_sheet.png"),
            TextureAtlas {
                layout: atlas_handle,
                index: 0,
            },
        ),
        Anchor(vec2(0.0, -0.5 + 3.0 / 18.0)),
        NormalMap::from_file("composite_sheet_normal.png", &asset_server),
        CompositeSprite {
            slices: vec![
                CompositeSpriteSlice {
                    index: 0,
                    offset: Vec2::new(10., 0.),
                },
                CompositeSpriteSlice {
                    index: 1,
                    offset: Vec2::new(-10., 0.),
                },
            ],
        },
        Transform::from_translation(vec3(0., -20., 20.)),
        Occluder2d::rectangle(12., 5.1),
        // component added to simulate height for the normal maps. Could be useful if the object is floating above the ground.
        // this can safely not be added, and it defaults to 0.
        SpriteHeight(0.),
    ));

    commands.spawn((
        Sprite::from_image(asset_server.load("lamp.png")),
        Anchor(vec2(0.0, -0.5 + 5.0 / 32.0)),
        Transform::from_translation(vec3(20., 0., 0.)),
        PointLight2d {
            intensity: 5.,
            range: 100.,
            color: Color::srgb(0.8, 0.8, 1.0),
            offset: vec3(0., 22., 0.),
            ..default()
        },
        LightHeight(22.),
    ));
}

// setting the sprite's z in relation to their y, so that Bevy's sprite renderer and firefly sort them properly.
fn z_sorting(mut sprites: Query<&mut Transform, With<Sprite>>) {
    for mut transform in &mut sprites {
        transform.translation.z = -transform.translation.y;
    }
}

#[derive(Resource, Default)]
struct Dragged(pub Option<Entity>);

fn drag_objects(
    mut objects: Query<(Entity, &mut Transform), With<Sprite>>,
    window: Single<&Window, With<PrimaryWindow>>,
    camera: Single<(&Camera, &GlobalTransform)>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut dragged: ResMut<Dragged>,
    mut gizmos: Gizmos,
) {
    let Some(cursor_position) = window
        .cursor_position()
        .and_then(|cursor| camera.0.viewport_to_world_2d(&camera.1, cursor).ok())
    else {
        dragged.0 = None;
        return;
    };

    if buttons.pressed(MouseButton::Left)
        && let Some(dragged) = dragged.0
        && let Ok((_, mut transform)) = objects.get_mut(dragged)
    {
        transform.translation.x = cursor_position.x;
        transform.translation.y = cursor_position.y;
        gizmos.circle_2d(
            Isometry2d::from_translation(transform.translation.xy()),
            3.,
            RED,
        );
        return;
    }

    if let Some((hovered, transform)) = objects.iter().min_by(|(_, a), (_, b)| {
        a.translation
            .xy()
            .distance(cursor_position)
            .total_cmp(&b.translation.xy().distance(cursor_position))
    }) && transform.translation.xy().distance(cursor_position) < 4.
    {
        gizmos.circle_2d(
            Isometry2d::from_translation(transform.translation.xy()),
            3.,
            RED,
        );
        if buttons.just_pressed(MouseButton::Left) {
            dragged.0 = Some(hovered);
        }
    }

    if !buttons.pressed(MouseButton::Left) {
        dragged.0 = None;
    }
}
