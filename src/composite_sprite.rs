use bevy::{ecs::component::Component, math::Vec2};

#[derive(Debug)]
pub struct CompositeSpriteSlice {
    pub offset: Vec2,
    pub index: usize,
}

#[derive(Component, Debug)]
pub struct CompositeSprite {
    pub slices: Vec<CompositeSpriteSlice>,
}
