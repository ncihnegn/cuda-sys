extern crate num;
#[macro_use]
extern crate num_derive;
extern crate num_traits;

pub mod runtime;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
