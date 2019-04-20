use std::io::{BufRead, BufReader, Read};
use std::collections::HashMap;
use itertools::Itertools;
use std::path::Path;
use std::fs::File;
use vulkano::device::{Device, Queue};
use std::sync::Arc;
use vulkano::pipeline::shader::{ShaderModule, GraphicsEntryPointAbstract, SpecializationConstants};
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::buffer::{CpuAccessibleBuffer, ImmutableBuffer, BufferUsage, TypedBufferAccess, BufferAccess};
use crate::model::ModelBuilderError::MissingMeshes;
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState, DrawIndexedError};
use vulkano::command_buffer::pool::standard::StandardCommandPoolBuilder;
use vulkano::SafeDeref;
use vulkano::descriptor::pipeline_layout::PipelineLayoutDesc;
use vulkano::pipeline::input_assembly::Index;
use vulkano::descriptor::descriptor_set::DescriptorSetsCollection;
use vulkano::pipeline::vertex::{VertexSource};
use std::marker::PhantomData;

impl From<tobj::Mesh> for Mesh<Vertex, u32> {
    fn from(mesh: tobj::Mesh) -> Mesh<Vertex, u32> {
        Mesh::from(&mesh)
    }
}

impl From<&tobj::Mesh> for Mesh<Vertex, u32> {
    fn from(mesh: &tobj::Mesh) -> Mesh<Vertex, u32> {
        assert_eq!(mesh.positions.len(), mesh.normals.len(), "need normals");

        let mut vertices = vec![];

        let indices = mesh.indices.clone();

        for i in 0..(mesh.positions.len() / 3) {
            vertices.push(Vertex {
                position: [
                    *mesh.positions.get(i * 3).unwrap(),
                    *mesh.positions.get((i * 3) + 1).unwrap(),
                    *mesh.positions.get((i * 3) + 2).unwrap(),
                ],
                normals: [
                    *mesh.normals.get(i * 3).unwrap(),
                    *mesh.normals.get((i * 3) + 1).unwrap(),
                    *mesh.normals.get((i * 3) + 2).unwrap(),
                ],
            });
        }

        Mesh {
            vertices,
            indices,
        }
    }
}

pub struct ModelBuilder<VertexDefinition, VertexType, IndexType, Layout, RenderP>
{
    queue:                Arc<Queue>,
    meshes:               Option<Vec<Mesh<VertexType, IndexType>>>,
    pipeline:             Arc<GraphicsPipeline<VertexDefinition, Layout, RenderP>>,
}

#[derive(Debug)]
pub enum ModelBuilderError {
    MissingMeshes,
    MissingVertexShader,
    MissingFragmentShader,
}

impl<VertexDefinition, VertexType, IndexType, Layout, RenderP> ModelBuilder<VertexDefinition, VertexType, IndexType, Layout, RenderP>
    where
        Mesh<VertexType, IndexType>: From<tobj::Mesh>,
        VertexType: Send + Sync + Clone,
        IndexType: Send + Sync + Clone,
{
    pub fn new(queue: Arc<Queue>, pipeline: Arc<GraphicsPipeline<VertexDefinition, Layout, RenderP>>) -> ModelBuilder<VertexDefinition, VertexType, IndexType, Layout, RenderP> {
        ModelBuilder {
            queue,
            pipeline,
            meshes:   None,
        }
    }

    pub fn with_obj_path(self, obj_path: &Path) -> Self {
        if let Ok(f) = File::open(obj_path) {
            self.with_obj(&mut BufReader::new(f))
        } else {
            self
        }
    }

    pub fn with_obj<A: BufRead>(self, obj: &mut A) -> Self {
        let mut result = tobj::load_obj_buf(obj, |_| Ok((Vec::new(), HashMap::new())));

        Self {
            meshes: result.map(|(objects, _)| objects.into_iter().map(|model| model.clone().mesh.into()).collect_vec()).ok(),
            ..self
        }
    }

    pub fn build(self) -> Result<Model<VertexDefinition, VertexType, IndexType, Layout, RenderP>, ModelBuilderError>
        where
            VertexType: Send + Sync + 'static,
            IndexType: Send + Sync + 'static,
    {
        if self.meshes.is_none() {
            return Err(MissingMeshes);
        }

        let meshes = self.meshes.unwrap();

        let vertices = meshes.clone().iter().flat_map(|mesh| mesh.clone().vertices.clone()).collect_vec();
        let indices = meshes.clone().iter().flat_map(|mesh| mesh.clone().indices.clone()).collect_vec();

        let vertex_buffer = ImmutableBuffer::from_iter(
            vertices.into_iter().clone(), BufferUsage::vertex_buffer(), self.queue.clone()
        ).unwrap();
        let index_buffer = ImmutableBuffer::from_iter(
            indices.into_iter().clone(), BufferUsage::index_buffer(), self.queue.clone(),
        ).unwrap();
        Ok(Model {
            vertex_buffer: vec![vertex_buffer.0.clone()],
            index_buffer: index_buffer.0.clone(),
            pipeline: self.pipeline.clone(),
            phantom: PhantomData::default(),
        })
    }
}

pub trait Drawable {
    fn draw<S: DescriptorSetsCollection>(&self, cmd_buf: AutoCommandBufferBuilder, dynamic_state: &DynamicState, sets: S) -> Result<AutoCommandBufferBuilder, DrawIndexedError>;
}

pub struct Model<VertexDefinition, VertexType, IndexType, Layout, RenderP> {
    pub vertex_buffer: Vec<Arc<BufferAccess + Send + Sync + 'static>>,
    pub index_buffer: Arc<TypedBufferAccess<Content = [IndexType]> + Sync + Send + 'static>,
    pub pipeline: Arc<GraphicsPipeline<VertexDefinition, Layout, RenderP>>,
    phantom: PhantomData<VertexType>,
}

impl<VertexDef, VertexType, IndexType, Layout, RenderP> Drawable for Model<VertexDef, VertexType, IndexType, Layout, RenderP>
    where
        Layout: Sync + Send + 'static,
        RenderP: Sync + Send + 'static,
        VertexDef: Sync + Send + 'static,
        IndexType: Index + Sized + Sync + Send + 'static,
        Arc<TypedBufferAccess<Content = [IndexType]>>: BufferAccess,
        GraphicsPipeline<VertexDef, Layout, RenderP>: GraphicsPipelineAbstract + VertexSource<(Vec<Arc<BufferAccess + Send + Sync>>)>,
{
    fn draw<S: DescriptorSetsCollection>(&self, cmd_buf: AutoCommandBufferBuilder, dynamic_state: &DynamicState, sets: S) -> Result<AutoCommandBufferBuilder, DrawIndexedError>
    {
        cmd_buf.draw_indexed(self.pipeline.clone(), dynamic_state, self.vertex_buffer.clone(), self.index_buffer.clone(), sets, ())
    }
}

pub trait RenderDrawable {
    type Error;

    fn draw_drawable<T: Drawable, S: DescriptorSetsCollection>(self, drawable: &T, dynamic_state: &DynamicState, sets: S) -> Result<Self, Self::Error>
        where
            Self: Sized;

}

impl RenderDrawable for AutoCommandBufferBuilder {
    type Error = DrawIndexedError;

    fn draw_drawable<T: Drawable, S: DescriptorSetsCollection>(self, drawable: &T, dynamic_state: &DynamicState, sets: S) -> Result<Self, Self::Error>
        where
            Self: Sized
    {
        drawable.draw(self, dynamic_state, sets)
    }

}


#[derive(Clone)]
pub struct Mesh<VertexDefinition, IndexDefinition> {
    pub vertices: Vec<VertexDefinition>,
    pub indices:  Vec<IndexDefinition>,
}

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normals: [f32; 3],
}

vulkano::impl_vertex!(Vertex, position, normals);
