use std::io::{BufRead, BufReader, Read};
use std::collections::HashMap;
use itertools::Itertools;
use core::borrow::{Borrow};
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
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::command_buffer::pool::standard::StandardCommandPoolBuilder;
use vulkano::SafeDeref;
use vulkano::descriptor::pipeline_layout::PipelineLayoutDesc;
use vulkano::pipeline::input_assembly::Index;

impl std::convert::From<&tobj::Mesh> for Mesh<Vertex, u32> {
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

pub struct ModelBuilder<VertexDefinition, IndexDefinition, Layout, RenderP>
{
    queue:                Arc<Queue>,
    meshes:               Option<Vec<Mesh<VertexDefinition, IndexDefinition>>>,
    pipeline:             Arc<GraphicsPipeline<VertexDefinition, Layout, RenderP>>,
}

pub enum ModelBuilderError {
    MissingMeshes,
    MissingVertexShader,
    MissingFragmentShader,
}

impl<VertexDefinition, IndexDefinition, Layout, RenderP> ModelBuilder<VertexDefinition, IndexDefinition, Layout, RenderP>
    where
        Mesh<VertexDefinition, IndexDefinition>: From<tobj::Mesh>,
        VertexDefinition: Send + Sync + Clone,
        IndexDefinition: Send + Sync + Clone,
{
    pub fn new(queue: Arc<Queue>, pipeline: Arc<GraphicsPipeline<VertexDefinition, Layout, RenderP>>) -> ModelBuilder<VertexDefinition, IndexDefinition, Layout, RenderP> {
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
        let result = tobj::load_obj_buf(obj, |_| Ok((Vec::new(), HashMap::new())));

        Self {
            meshes: result.map(|(mut objs, _)| objs.iter_mut().map(|model| model.mesh.into()).collect_vec()).ok(),
            ..self
        }
    }

    pub fn build(mut self) -> Result<Model<VertexDefinition, IndexDefinition, Layout, RenderP>, ModelBuilderError> {
        if self.meshes.is_none() {
            return Err(MissingMeshes);
        }

        let meshes = self.meshes.unwrap();

        let vertices = meshes.clone().iter().flat_map(|mesh| mesh.clone().vertices.clone()).collect_vec();
        let indices = meshes.clone().iter().flat_map(|mesh| mesh.clone().indices.clone()).collect_vec();

        let vertex_buffer = ImmutableBuffer::from_iter(
            vertices.into_iter(), BufferUsage::vertex_buffer(), self.queue.clone()
        ).unwrap();
        let index_buffer = ImmutableBuffer::from_iter(
            indices.into_iter(), BufferUsage::index_buffer(), self.queue.clone(),
        ).unwrap();
        Ok(Model {
            vertex_buffer: vertex_buffer.0.clone(),
            index_buffer: index_buffer.0.clone(),
            pipeline: self.pipeline.clone(),
        })
    }
}

trait Drawable {
    type Pipeline;
    fn draw(&self, cmd_buf: AutoCommandBufferBuilder) -> AutoCommandBufferBuilder ;
}

pub struct Model<VertexDefinition, IndexDefinition, Layout, RenderP>
{
    pub vertex_buffer: Arc<BufferAccess + Send + Sync>,
    pub index_buffer: Arc<TypedBufferAccess<Content = [IndexDefinition]>>,
    pub pipeline: Arc<GraphicsPipeline<VertexDefinition, Layout, RenderP>>,
}

impl<VertexDefinition, IndexDefinition, Layout, RenderP> Drawable for Model<VertexDefinition, IndexDefinition, Layout, RenderP> where
    IndexDefinition: Index,
    VertexDefinition: Sync + Send + Clone + BufferAccess,
    TypedBufferAccess<Content = [IndexDefinition]>: Sync + Send + BufferAccess,
    Layout: PipelineLayoutAbstract + SafeDeref + Sync + Send,
    RenderP: RenderPassAbstract + SafeDeref + Sync + Send,
    GraphicsPipeline<VertexDefinition, Layout, RenderP>: GraphicsPipelineAbstract,
{
    type Pipeline = Arc<GraphicsPipeline<VertexDefinition, Layout, RenderP>>;

    fn draw(&self, cmd_buf: AutoCommandBufferBuilder<StandardCommandPoolBuilder>) -> AutoCommandBufferBuilder {
        cmd_buf.draw_indexed(self.pipeline.clone(), &DynamicState::default(), vec![self.vertex_buffer.clone()].clone(), self.index_buffer.clone(), (), ()).unwrap()
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
