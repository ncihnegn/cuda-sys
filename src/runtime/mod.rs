#[allow(dead_code)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
#[allow(clippy::unreadable_literal)]
mod bind;

#[macro_use]
pub mod data_type;

use std::ffi::c_void;
use std::ptr;

use num::ToPrimitive;

use self::bind::*;
use self::data_type::*;

pub fn get_device_count() -> Result<i32, ErrorCode> {
    let mut device_count: i32 = 0;
    check_error!(
        unsafe { cudaGetDeviceCount(&mut device_count) },
        device_count
    )
}

pub fn set_device(id: i32) -> Result<(), ErrorCode> {
    check_error!(unsafe { cudaSetDevice(id) }, ())
}

pub fn get_device_prop(id: i32) -> Result<DeviceProp, ErrorCode> {
    let mut prop = cudaDeviceProp::default();
    check_error!(
        unsafe { cudaGetDeviceProperties(&mut prop, id) },
        DeviceProp::from(&prop)
    )
}

pub enum Component {
    Driver,
    Runtime,
}

fn get_version(d: &Component) -> Result<Version, ErrorCode> {
    let mut version: i32 = 0;
    let e = match d {
        Component::Driver => unsafe { cudaDriverGetVersion(&mut version) },
        Component::Runtime => unsafe { cudaRuntimeGetVersion(&mut version) },
    };
    check_error!(e, Version::from_i32(version))
}
pub fn driver_get_version() -> Result<Version, ErrorCode> {
    get_version(&Component::Driver)
}

pub fn runtime_get_version() -> Result<Version, ErrorCode> {
    get_version(&Component::Runtime)
}

pub fn malloc(size: usize) -> Result<*mut c_void, ErrorCode> {
    let mut pt: *mut c_void = ptr::null_mut();
    check_error!(unsafe { cudaMalloc(&mut pt, size) }, pt)
}

pub fn memcpy(
    dst: &mut c_void,
    src: &c_void,
    count: usize,
    kind: &MemcpyKind,
) -> Result<(), ErrorCode> {
    let default = MemcpyKind::Default.to_u32().unwrap();
    let kc = kind.to_u32().unwrap_or(default);
    check_error!(unsafe { cudaMemcpy(dst, src, count, kc) }, ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malloc_test() {
        assert!(malloc(1).is_ok(),);
    }

    #[test]
    fn memcpy_d2d_test() {
        let count = 1;
        let dptr0 = malloc(count).unwrap();
        let dptr1 = malloc(count).unwrap();
        let dref0 = unsafe { &mut *dptr0 };
        let dref1 = unsafe { &*dptr1 };
        assert!(memcpy(dref0, dref1, count, &MemcpyKind::Default).is_ok());
    }
}
