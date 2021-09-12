use nalgebra::DVector;

// Left and right bound for an interval.
// By convention, the right boundary is excluded, i.e. [min, max).
pub struct Bounds<T> {
    min: T,
    max: T,
}

// Describes continuous dimension withing a state space that should be tiled.
pub struct ContinuousDimension {
    bounds: Bounds<f64>,
    step_count: usize,
}

// Describes a tile partition of a continuous dimension.
struct ContinuousPartition {
    origin: f64,
    step_size: f64,
    step_count: usize,
}

// Describes a tile partition of an integer dimension.
struct IntegerPartition {
    origin: i32,
    step_count: usize,
}

// One tiling of a state space.
struct Tiling {
    continuous_partitions: Vec<ContinuousPartition>,
    integer_partitions: Vec<IntegerPartition>,
    tile_count: usize,
}

// Set of tilings of a state space.
pub struct TilingSet {
    tilings: Vec<Tiling>,
}

impl<T> Bounds<T> {
    pub fn new(min: T, max: T) -> Self {
        Bounds { min: min, max: max }
    }
}

impl ContinuousDimension {
    pub fn new(min: f64, max: f64, step_count: usize) -> Self {
        ContinuousDimension {
            bounds: Bounds::new(min, max),
            step_count: step_count,
        }
    }
}

impl Tiling {
    fn from_dimensions_and_origin(
        continuous_dimensions: &Vec<ContinuousDimension>,
        integer_dimensions: &Vec<Bounds<i32>>,
        origin: &DVector<f64>,
    ) -> Self {
        let c_len = continuous_dimensions.len();
        let i_len = integer_dimensions.len();

        let step_size = DVector::from_iterator(
            c_len,
            continuous_dimensions
                .iter()
                .map(|d| (d.bounds.max - d.bounds.min) / (d.step_count) as f64),
        );

        let continuous_partitions: Vec<ContinuousPartition> = (0..c_len)
                .map(|i| ContinuousPartition {
                    origin: origin[i],
                    step_size: step_size[i],
                    step_count: continuous_dimensions[i].step_count,
                })
                .collect();
        
        let integer_partitions: Vec<IntegerPartition> = (0..i_len)
                .map(|i| IntegerPartition {
                    origin: integer_dimensions[i].min,
                    step_count: (integer_dimensions[i].max - integer_dimensions[i].min) as usize,
                })
                .collect();
        
        let tile_count: usize = continuous_partitions
            .iter()
            .fold(1, |acc, p| acc * p.step_count)
            *integer_partitions
                .iter()
                .fold(1, |acc, p| acc * p.step_count);

        Tiling {
            continuous_partitions: continuous_partitions,
            integer_partitions: integer_partitions,
            tile_count: tile_count,
        }
    }

    // Returns an index of a tile containing the given point in the state space.
    fn get_tile(&self, pc: &DVector<f64>, pi: &DVector<i32>) -> usize {
        assert_eq!(pc.len(), self.continuous_partitions.len());
        assert_eq!(pi.len(), self.integer_partitions.len());

        let mut offset = 0;

        // By convention, assume feature layout to be such that when we iterate
        // over features, the coordinate in the first dimension (which is first
        // continuous dimension) changes the fastest and the cordinate in the
        // last dimension (which is the last integer dimension) changes the
        // slowest.

        for i in (0..pi.len()).rev() {
            let p = &self.integer_partitions[i];
            let x = pi[i];
            let index = ((x - p.origin).max(0) as usize).min(p.step_count - 1);
            offset = offset * p.step_count + index;
        }

        for i in (0..pc.len()).rev() {
            let p = &self.continuous_partitions[i];
            let x = pc[i];
            let step = ((x - p.origin) / p.step_size).max(0.0);
            let index = (step as usize).min(p.step_count - 1);
            offset = offset * p.step_count + index;
        }

        offset
    }
}

impl TilingSet {
    // Creates a tiling set for a N + M dimension space, where N is the number
    // of continuous dimensions and M is the number of integer ones.
    pub fn from_dimensions(
        continuous_dimensions: &Vec<ContinuousDimension>,
        integer_dimensions: &Vec<Bounds<i32>>,
        count: usize,
    ) -> Self {
        let c_len = continuous_dimensions.len();

        let mut tilings = Vec::new();
        let mut origin =
            DVector::from_iterator(c_len, continuous_dimensions.iter().map(|d| d.bounds.min));
        let step_size = DVector::from_iterator(
            c_len,
            continuous_dimensions
                .iter()
                .map(|d| (d.bounds.max - d.bounds.min) / (d.step_count) as f64),
        );
        let offset_step =
            DVector::from_iterator(c_len, (0..c_len).map(|i| step_size[i] / count as f64));

        for _ in 0..count {
            tilings.push(Tiling::from_dimensions_and_origin(
                continuous_dimensions,
                integer_dimensions,
                &origin,
            ));

            origin += &offset_step;
        }

        TilingSet { tilings: tilings }
    }

    // Returns number of tilings.
    pub fn count(&self) -> usize {
        self.tilings.len()
    }

    // Returns the total number of features.
    pub fn tile_count(&self) -> usize {
        self.tilings.iter().map(|t| t.tile_count).sum()
    }

    // Returns the indices of tiles (across all tilings, one tile per tiling)
    // that contain the given point in the state space.
    // Dimension of the return vector is equal to the number of tilings, or count().
    pub fn get_tiles(&self, pc: &DVector<f64>, pi: &DVector<i32>) -> Vec<usize> {
        let mut feature_indices = Vec::with_capacity(self.tilings.len());
        let mut index_offset = 0;
        for t in &self.tilings {
            feature_indices.push(t.get_tile(pc, pi) + index_offset);
            index_offset += t.tile_count;
        }
        feature_indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pc(items: &[f64]) -> DVector<f64> {
        DVector::from_iterator(items.len(), items.iter().map(|d| *d))
    }

    fn pi(items: &[i32]) -> DVector<i32> {
        DVector::from_iterator(items.len(), items.iter().map(|d| *d))
    }

    #[test]
    fn single_tiling_1d() {
        let c1 = ContinuousDimension::new(0.0, 10.0, 10);
        let tilings = TilingSet::from_dimensions(&vec![c1], &Vec::new(), 1);

        assert_eq!(tilings.count(), 1);
        assert_eq!(tilings.tile_count(), 10);

        assert_eq!(tilings.get_tiles(&pc(&[0.0]), &pi(&[])), vec![0]);
        assert_eq!(tilings.get_tiles(&pc(&[-1.0]), &pi(&[])), vec![0]);
        assert_eq!(tilings.get_tiles(&pc(&[9.5]), &pi(&[])), vec![9]);
        assert_eq!(tilings.get_tiles(&pc(&[11.0]), &pi(&[])), vec![9]);
    }

    #[test]
    fn single_tiling_2d() {
        let c1 = ContinuousDimension::new(-10.0, 10.0, 20);
        let c2 = ContinuousDimension::new(0.0, 10.0, 10);
        let tilings = TilingSet::from_dimensions(&vec![c1, c2], &Vec::new(), 1);

        assert_eq!(tilings.count(), 1);
        assert_eq!(tilings.tile_count(), 200);

        assert_eq!(tilings.get_tiles(&pc(&[-1.0, 0.0]), &pi(&[])), vec![9]);
        assert_eq!(tilings.get_tiles(&pc(&[-1.0, 1.0]), &pi(&[])), vec![29]);
    }

    #[test]
    fn single_tiling_1i() {
        let i1 = Bounds::new(0, 5);
        let tilings = TilingSet::from_dimensions(&vec![], &vec![i1], 1);

        assert_eq!(tilings.count(), 1);
        assert_eq!(tilings.tile_count(), 5);

        assert_eq!(tilings.get_tiles(&pc(&[]), &pi(&[0])), vec![0]);
        assert_eq!(tilings.get_tiles(&pc(&[]), &pi(&[-1])), vec![0]);
        assert_eq!(tilings.get_tiles(&pc(&[]), &pi(&[1])), vec![1]);
        assert_eq!(tilings.get_tiles(&pc(&[]), &pi(&[4])), vec![4]);
        assert_eq!(tilings.get_tiles(&pc(&[]), &pi(&[5])), vec![4]);
    }

    #[test]
    fn single_tiling_1d_1i() {
        let c1 = ContinuousDimension::new(0.0, 10.0, 10);
        let i1 = Bounds::new(0, 5);
        let tilings = TilingSet::from_dimensions(&vec![c1], &vec![i1], 1);

        assert_eq!(tilings.count(), 1);
        assert_eq!(tilings.tile_count(), 50);

        assert_eq!(tilings.get_tiles(&pc(&[0.0]), &pi(&[0])), vec![0]);
        assert_eq!(tilings.get_tiles(&pc(&[0.0]), &pi(&[1])), vec![10]);
    }

    #[test]
    fn multi_tiling_1d_1i() {
        let c1 = ContinuousDimension::new(0.0, 10.0, 10);
        let i1 = Bounds::new(0, 5);
        let tilings = TilingSet::from_dimensions(&vec![c1], &vec![i1], 3);

        assert_eq!(tilings.count(), 3);
        assert_eq!(tilings.tile_count(), 150);

        // Point (0, 0) should be in tile 0 on all tilings.
        assert_eq!(
            tilings.get_tiles(&pc(&[0.0]), &pi(&[0])),
            vec![0, 50, 100]
        );

        // Offset step is 1.0/3, so point 1.4 should be on tile 1 for tilings
        // 0 and 1, but tile 0 on tiling 2.
        assert_eq!(
            tilings.get_tiles(&pc(&[1.4]), &pi(&[0])),
            vec![1, 51, 100]
        );
    }
}
