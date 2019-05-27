use cgmath::{Deg, Vector3, Quaternion, One, RelativeEq, Rotation3,  BaseFloat, Matrix4, vec3};

#[derive(Debug)]
pub struct Camera<T> {
    position:    Vector3<T>,
    orientation: Quaternion<T>,
}

impl<T> Camera<T>
    where
        T: RelativeEq,
        T: BaseFloat,
{

    pub fn pitch(&mut self, pitch: Deg<T>) {
        self.orientation = Quaternion::from_axis_angle(
            vec3(T::one(), T::zero(), T::zero()), pitch
        ) * self.orientation;
    }

    pub fn turn(&mut self, yaw: Deg<T>) {
        self.orientation = Quaternion::from_axis_angle(
            self.orientation * vec3(T::zero(), T::one(), T::zero()), yaw
        ) * self.orientation;
    }

    pub fn move_forward(&mut self, movement: T) {
        self.position -= self.forward() * movement;
    }

    pub fn move_left(&mut self, movement: T) {
        self.position -= self.left() * movement;
    }

    pub fn move_up(&mut self, movement: T) {
        self.position += self.up() * movement;
    }

    fn forward(&self) -> Vector3<T> {
        self.orientation.conjugate() * vec3(T::zero(), T::zero(), T::one())
    }

    fn left(&self) -> Vector3<T> {
        self.orientation.conjugate() * vec3(T::one(), T::zero(), T::zero())
    }

    fn up(&self) -> Vector3<T> {
        self.orientation.conjugate() * vec3(T::zero(), T::one(), T::zero())
    }

    pub fn position(&self) -> &Vector3<T> {
        &self.position
    }

    pub fn view_matrix(&self) -> Matrix4<T> {
        Matrix4::from(self.orientation) * Matrix4::from_translation(-self.position)
    }

}

impl Default for Camera<f32> {
    fn default() -> Self {
        Camera {
            position:    Vector3::new(0.0, 0.0, -1.0),
            orientation: Quaternion::one(),
        }
    }
}
