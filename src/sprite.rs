use std::any::TypeId;

use bevy::{
    asset::AsAssetId,
    camera::visibility::{self, VisibilityClass},
    ecs::{lifecycle::HookContext, world::DeferredWorld},
    prelude::*,
    sprite::Anchor,
};

pub fn add_sprite_visibility_classes(
    mut world: DeferredWorld<'_>,
    HookContext { entity, .. }: HookContext,
) {
    if let Some(mut visibility_class) = world.get_mut::<VisibilityClass>(entity) {
        visibility_class.push(TypeId::of::<Sprite>());
        visibility_class.push(TypeId::of::<FireflySprite>());
    }
}

/// Describes a sprite to be rendered to a 2D camera
#[derive(Component, Debug, Default, Clone, Reflect)]
#[require(Transform, Visibility, VisibilityClass, Anchor)]
#[reflect(Component, Default, Debug, Clone)]
#[component(on_add = add_sprite_visibility_classes)]
pub struct FireflySprite {
    /// The image used to render the sprite
    pub image: Handle<Image>,
    /// The (optional) texture atlas used to render the sprite
    pub texture_atlas: Option<TextureAtlas>,
    /// The sprite's color tint
    pub color: Color,
    /// Flip the sprite along the `X` axis
    pub flip_x: bool,
    /// Flip the sprite along the `Y` axis
    pub flip_y: bool,
    /// An optional custom size for the sprite that will be used when rendering, instead of the size
    /// of the sprite's image
    pub custom_size: Option<Vec2>,
    /// An optional rectangle representing the region of the sprite's image to render, instead of rendering
    /// the full image. This is an easy one-off alternative to using a [`TextureAtlas`].
    ///
    /// When used with a [`TextureAtlas`], the rect
    /// is offset by the atlas's minimal (top-left) corner position.
    pub rect: Option<Rect>,
    /// How the sprite's image will be scaled.
    pub image_mode: FireflySpriteImageMode,
}

impl FireflySprite {
    /// Create a Sprite with a custom size
    pub fn sized(custom_size: Vec2) -> Self {
        FireflySprite {
            custom_size: Some(custom_size),
            ..Default::default()
        }
    }

    /// Create a sprite from an image
    pub fn from_image(image: Handle<Image>) -> Self {
        Self {
            image,
            ..Default::default()
        }
    }

    /// Create a sprite from an image, with an associated texture atlas
    pub fn from_atlas_image(image: Handle<Image>, atlas: TextureAtlas) -> Self {
        Self {
            image,
            texture_atlas: Some(atlas),
            ..Default::default()
        }
    }

    /// Create a sprite from a solid color
    pub fn from_color(color: impl Into<Color>, size: Vec2) -> Self {
        Self {
            color: color.into(),
            custom_size: Some(size),
            ..Default::default()
        }
    }

    /// Computes the pixel point where `point_relative_to_sprite` is sampled
    /// from in this sprite. `point_relative_to_sprite` must be in the sprite's
    /// local frame. Returns an Ok if the point is inside the bounds of the
    /// sprite (not just the image), and returns an Err otherwise.
    pub fn compute_pixel_space_point(
        &self,
        point_relative_to_sprite: Vec2,
        anchor: Anchor,
        images: &Assets<Image>,
        texture_atlases: &Assets<TextureAtlasLayout>,
    ) -> Result<Vec2, Vec2> {
        let image_size = images
            .get(&self.image)
            .map(Image::size)
            .unwrap_or(UVec2::ONE);

        let atlas_rect = self
            .texture_atlas
            .as_ref()
            .and_then(|s| s.texture_rect(texture_atlases))
            .map(|r| r.as_rect());
        let texture_rect = match (atlas_rect, self.rect) {
            (None, None) => Rect::new(0.0, 0.0, image_size.x as f32, image_size.y as f32),
            (None, Some(sprite_rect)) => sprite_rect,
            (Some(atlas_rect), None) => atlas_rect,
            (Some(atlas_rect), Some(mut sprite_rect)) => {
                // Make the sprite rect relative to the atlas rect.
                sprite_rect.min += atlas_rect.min;
                sprite_rect.max += atlas_rect.min;
                sprite_rect
            }
        };

        let sprite_size = self.custom_size.unwrap_or_else(|| texture_rect.size());
        let sprite_center = -anchor.as_vec() * sprite_size;

        let mut point_relative_to_sprite_center = point_relative_to_sprite - sprite_center;

        if self.flip_x {
            point_relative_to_sprite_center.x *= -1.0;
        }
        // Texture coordinates start at the top left, whereas world coordinates start at the bottom
        // left. So flip by default, and then don't flip if `flip_y` is set.
        if !self.flip_y {
            point_relative_to_sprite_center.y *= -1.0;
        }

        if sprite_size.x == 0.0 || sprite_size.y == 0.0 {
            return Err(point_relative_to_sprite_center);
        }

        let sprite_to_texture_ratio = {
            let texture_size = texture_rect.size();
            Vec2::new(
                texture_size.x / sprite_size.x,
                texture_size.y / sprite_size.y,
            )
        };

        let point_relative_to_texture =
            point_relative_to_sprite_center * sprite_to_texture_ratio + texture_rect.center();

        // TODO: Support `SpriteImageMode`.

        if texture_rect.contains(point_relative_to_texture) {
            Ok(point_relative_to_texture)
        } else {
            Err(point_relative_to_texture)
        }
    }
}

impl From<Handle<Image>> for FireflySprite {
    fn from(image: Handle<Image>) -> Self {
        Self::from_image(image)
    }
}

impl AsAssetId for FireflySprite {
    type Asset = Image;

    fn as_asset_id(&self) -> AssetId<Self::Asset> {
        self.image.id()
    }
}

#[derive(Default, Debug, Clone, Reflect, PartialEq)]
#[reflect(Debug, Default, Clone)]
/// A sprite instance is rendered from a texture atlas
pub struct SpriteInstance {
    pub index: usize,
    pub offset: Vec2,
}

/// Controls how the image is altered when scaled.
#[derive(Default, Debug, Clone, Reflect, PartialEq)]
#[reflect(Debug, Default, Clone)]
pub enum FireflySpriteImageMode {
    /// The sprite will take on the size of the image by default, and will be stretched or shrunk if [`Sprite::custom_size`] is set.
    #[default]
    Auto,
    /// The texture will be scaled to fit the rect bounds defined in [`Sprite::custom_size`].
    /// Otherwise no scaling will be applied.
    Scale(SpriteScalingMode),
    /// The texture will be cut in 9 slices, keeping the texture in proportions on resize
    Sliced(TextureSlicer),
    /// The texture will be repeated if stretched beyond `stretched_value`
    Tiled {
        /// Should the image repeat horizontally
        tile_x: bool,
        /// Should the image repeat vertically
        tile_y: bool,
        /// The texture will repeat when the ratio between the *drawing dimensions* of texture and the
        /// *original texture size* are above this value.
        stretch_value: f32,
    },
    /// The texture will be rendered by the manually configured sprite instances in this vector.
    Instances(Vec<SpriteInstance>),
}

impl FireflySpriteImageMode {
    /// Returns true if this mode uses slices internally ([`SpriteImageMode::Sliced`] or [`SpriteImageMode::Tiled`])
    #[inline]
    pub fn uses_slices(&self) -> bool {
        matches!(
            self,
            Self::Sliced(..) | Self::Tiled { .. } | Self::Instances(..)
        )
    }

    /// Returns [`SpriteScalingMode`] if scale is presented or [`Option::None`] otherwise.
    #[inline]
    #[must_use]
    pub const fn scale(&self) -> Option<SpriteScalingMode> {
        if let Self::Scale(scale) = self {
            Some(*scale)
        } else {
            None
        }
    }
}
