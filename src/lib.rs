use std::io::{BufRead, BufReader};
use std::collections::HashMap;
use itertools::Itertools;
use core::borrow::{Borrow};
use std::path::Path;
use std::fs::File;
use fuji::Fuji;
use crate::model::{Model, ModelBuilder, Vertex, Mesh};
use vulkano::pipeline::GraphicsPipeline;
use std::sync::Arc;

mod model;

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
            fuji: self.fuji
        }
    }
}

pub struct FrozenGameInstance {
    fuji: Fuji,
}

impl FrozenGameInstance {
    pub fn build_model<VD, ID, L, RP>(&self, pipeline: Arc<GraphicsPipeline<VD, L, RP>>) -> ModelBuilder<VD, ID, L, RP>
        where
            Mesh<VD, ID>: From<tobj::Mesh> + Clone + Sync + Send,
            VD: Clone + Sync + Send,
            ID: Clone + Sync + Send,
    {
        ModelBuilder::new(self.fuji.graphics_queue().clone(), pipeline)
    }
}
