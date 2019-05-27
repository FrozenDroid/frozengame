





use fuji::Fuji;
use crate::model::{ModelBuilder, Mesh};
use vulkano::pipeline::GraphicsPipeline;
use std::sync::Arc;
use vulkano::pipeline::vertex::{Vertex};


pub mod model;
pub mod camera;

pub struct FrozenGameBuilder {
    fuji: Fuji,
}

impl FrozenGameBuilder {
    pub fn new(fuji: Fuji) -> Self {
        FrozenGameBuilder {
            fuji,
        }
    }

    pub fn build(self) -> FrozenGameInstance {
        FrozenGameInstance {
            fuji: Arc::new(self.fuji)
        }
    }
}

pub struct FrozenGameInstance {
    pub fuji: Arc<Fuji>,
}

impl FrozenGameInstance {
    pub fn build_model<VD, VT, IT, L, RP>(&self, pipeline: Arc<GraphicsPipeline<VD, L, RP>>) -> ModelBuilder<VD, VT, IT, L, RP>
        where
            Arc<GraphicsPipeline<VT, L, RP>>: Clone,
            Mesh<VT, IT>: From<tobj::Mesh> + Clone + Sync + Send,
            IT: Clone + Sync + Send,
            VT: Clone + Sync + Send + Vertex
    {
        ModelBuilder::new(self.fuji.graphics_queue().clone(), pipeline)
    }
}
