#[no_mangle]
pub extern "C" fn mesh_route(x: i32, y: i32, dst_x: i32, dst_y: i32) -> i32 {
    if dst_x < x { 4 } else if dst_x > x { 3 } else if dst_y < y { 1 } else if dst_y > y { 2 } else { 0 }
}
