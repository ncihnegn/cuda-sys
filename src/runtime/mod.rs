#[allow(dead_code)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
#[allow(clippy::unreadable_literal)]
mod bind;

#[macro_use]
pub mod data_type;

use self::bind::*;
use self::data_type::*;

pub fn get_device_count() -> Result<i32, ErrorCode> {
    let mut device_count: i32 = 0;
    let e = ErrorCode::check(unsafe { cudaGetDeviceCount(&mut device_count) });
    check_error!(e, device_count)
}

pub fn set_device(id: i32) -> Result<(), ErrorCode> {
    let e = ErrorCode::check(unsafe { cudaSetDevice(id) });
    check_error!(e, ())
}

pub fn get_device_prop(id: i32) -> Result<DeviceProp, ErrorCode> {
    let mut prop = cudaDeviceProp::default();
    let e = ErrorCode::check(unsafe { cudaGetDeviceProperties(&mut prop, id) });
    check_error!(e, DeviceProp::from(&prop))
}

pub enum Component {
    Driver,
    Runtime,
}

fn get_version(d: &Component) -> Result<Version, ErrorCode> {
    let mut version: i32 = 0;
    let e = ErrorCode::check(match d {
        Component::Driver => unsafe { cudaDriverGetVersion(&mut version) },
        Component::Runtime => unsafe { cudaRuntimeGetVersion(&mut version) },
    });
    check_error!(e, Version::from_i32(version))
}
pub fn driver_get_version() -> Result<Version, ErrorCode> {
    get_version(&Component::Driver)
}

pub fn runtime_get_version() -> Result<Version, ErrorCode> {
    get_version(&Component::Runtime)
}
