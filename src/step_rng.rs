use rand::Rng;
use std::convert::Infallible;

/// This generates an arithmetic sequence
/// (i.e. adds a constant each step) over a u64 number,
/// using wrapping arithmetic.
///
/// If the increment is 0 the generator yields a constant.
///
/// This was previously a part of the `rand` crate.
///
/// See for documentation:
/// <https://people.eecs.berkeley.edu/~pschafhalter/pub/erdos/doc/rand/rngs/mock/struct.StepRng.html>
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepRng {
    initial: u64,
    increment: u64,
    next: u64,
}

impl StepRng {
    pub fn new(initial: u64, increment: u64) -> Self {
        StepRng {
            initial,
            increment,
            next: initial,
        }
    }
}

impl rand::TryRng for StepRng {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        let current = self.next as u32;
        self.next = self.next.wrapping_add(self.increment);
        Ok(current)
    }

    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        let current = self.next;
        self.next = self.next.wrapping_add(self.increment);
        Ok(current)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Infallible> {
        for byte in dest.iter_mut() {
            *byte = self.next_u64() as u8;
        }
        Ok(())
    }
}
