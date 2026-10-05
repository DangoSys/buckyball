//! Transaction facts from real handshakes. No hardware cursor or phase FSM is mirrored.
use std::ffi::c_void;
#[derive(Clone, Copy)]
struct Access { base: u64, bytes: u64, write: bool }
struct Transaction {
    id: u32, age: u64, sealed: bool, ranges: Vec<Access>,
    pre: Vec<bool>, post: Vec<bool>, granted: bool, dma_ack: bool,
}
struct Model {
    capacity: usize, address_bits: u32, line_bytes: u64, max_ranges: usize,
    age: u64, live: Vec<Transaction>, maintenance: Option<(u32, u32, usize)>,
}
impl Model {
    fn transaction(&self,id:u32)->&Transaction { self.live.iter().find(|t|t.id==id).expect("unknown transaction") }
    fn transaction_mut(&mut self,id:u32)->&mut Transaction { self.live.iter_mut().find(|t|t.id==id).expect("unknown transaction") }
    fn overlap(a:Access,b:Access)->bool { a.base<b.base+b.bytes && b.base<a.base+a.bytes }
    fn bounds(t:&Transaction)->Option<Access> {
        t.ranges.first().map(|first| {
            let base=t.ranges.iter().map(|a|a.base).min().unwrap();
            let end=t.ranges.iter().map(|a|a.base+a.bytes).max().unwrap();
            Access{base,bytes:end-base,write:first.write}
        })
    }
    fn older_conflict(&self,t:&Transaction)->bool {
        let a=Self::bounds(t).expect("memory grant without ranges");
        self.live.iter().filter(|old|old.age<t.age).any(|old| {
            !old.sealed || Self::bounds(old).is_some_and(|b|Self::overlap(a,b)&&(a.write||b.write))
        })
    }
}
unsafe fn model<'a>(p:*mut c_void)->&'a mut Model { &mut *p.cast::<Model>() }
#[no_mangle]
pub extern "C" fn interlock_ref_create(entries:u32,address_bits:u32,line_bytes:u32,max_ranges:u32)->*mut c_void {
    assert!(entries>0&&address_bits<64&&line_bytes.is_power_of_two()&&max_ranges>0);
    Box::into_raw(Box::new(Model{capacity:entries as usize,address_bits,line_bytes:line_bytes as u64,max_ranges:max_ranges as usize,age:0,live:Vec::new(),maintenance:None})).cast()
}
#[no_mangle]
pub unsafe extern "C" fn interlock_ref_destroy(p:*mut c_void) { drop(Box::from_raw(p.cast::<Model>())); }
#[no_mangle]
pub unsafe extern "C" fn interlock_ref_reset(p:*mut c_void) { let m=model(p);m.live.clear();m.maintenance=None;m.age=0; }
#[no_mangle]
pub unsafe extern "C" fn interlock_ref_live(p:*mut c_void)->u32 { model(p).live.len() as u32 }
#[no_mangle]
pub unsafe extern "C" fn interlock_ref_reserve(p:*mut c_void,id:u32) {
    let m=model(p);assert!(m.live.len()<m.capacity&&id<256&&m.live.iter().all(|t|t.id!=id));m.age+=1;
    m.live.push(Transaction{id,age:m.age,sealed:false,ranges:Vec::new(),pre:Vec::new(),post:Vec::new(),granted:false,dma_ack:false});
}
#[no_mangle]
pub unsafe extern "C" fn interlock_ref_info(p:*mut c_void,id:u32,has_mem:u8,base:u64,bytes:u64,write:u8,last:u8) {
    let m=model(p);let line=m.line_bytes;let capacity=m.max_ranges;
    if has_mem!=0 { assert!(bytes>0&&base.checked_add(bytes).is_some_and(|end|end<=(1<<m.address_bits))); }
    let t=m.transaction_mut(id);assert!(!t.sealed);
    if has_mem==0 { assert!(t.ranges.is_empty()&&last!=0);t.sealed=true;return; }
    assert!(t.ranges.len()<capacity&&(t.ranges.len()+1<capacity||last!=0));
    assert!(t.ranges.first().is_none_or(|a|a.write==(write!=0)));
    let first=base&!(line-1);let end=(base+bytes-1)|(line-1);
    t.ranges.push(Access{base:first,bytes:end-first+1,write:write!=0});t.pre.push(false);t.post.push(false);
    t.sealed=last!=0;
}
#[no_mangle]
pub unsafe extern "C" fn interlock_ref_cancel(p:*mut c_void,id:u32) {
    let m=model(p);assert!(!m.transaction(id).sealed,"cancel after seal");m.live.retain(|t|t.id!=id);
}
#[no_mangle]
pub unsafe extern "C" fn interlock_ref_cpu_allow(p:*mut c_void,addr:u64,bytes:u32,write:u8,older:u8,reserve_now:u8)->u8 {
    let m=model(p);assert!(bytes.is_power_of_two()&&bytes<=8&&addr%bytes as u64==0);
    assert!(addr/m.line_bytes==(addr+bytes as u64-1)/m.line_bytes);
    if older!=0||reserve_now!=0{return 0;}
    let cpu=Access{base:addr,bytes:bytes as u64,write:write!=0};
    u8::from(!m.live.iter().any(|t|!t.sealed||t.ranges.iter().any(|&a|Model::overlap(cpu,a)&&(cpu.write||a.write))))
}
#[no_mangle]
pub unsafe extern "C" fn interlock_ref_maint_offer(p:*mut c_void,id:u32,op:u32,first:u64,last:u64) {
    let m=model(p);assert!(m.maintenance.is_none());let t=m.transaction(id);assert!(t.sealed&&!t.ranges.is_empty());
    let write=t.ranges[0].write;
    if op==2 { assert!(write&&t.granted&&t.dma_ack); }
    else { assert!(op==u32::from(write)&&!t.granted&&!m.older_conflict(t)); }
    let done=if op==2{&t.post}else{&t.pre};
    let index=t.ranges.iter().enumerate().position(|(i,a)|!done[i]&&a.base==first&&a.base+a.bytes-m.line_bytes==last).expect("wrong or duplicate maintenance extent");
    m.maintenance=Some((id,op,index));
}
#[no_mangle]
pub unsafe extern "C" fn interlock_ref_maint_ack(p:*mut c_void,id:u32) {
    let m=model(p);let (owner,op,index)=m.maintenance.take().expect("ACK without maintenance request");assert_eq!(owner,id);
    let t=m.transaction_mut(id);if op==2{t.post[index]=true;}else{t.pre[index]=true;}
}
#[no_mangle]
pub unsafe extern "C" fn interlock_ref_grant(p:*mut c_void,id:u32) {
    let m=model(p);let t=m.transaction(id);assert!(t.sealed&&!t.ranges.is_empty()&&t.pre.iter().all(|b|*b)&&!t.granted&&!m.older_conflict(t));
    m.transaction_mut(id).granted=true;
}
#[no_mangle]
pub unsafe extern "C" fn interlock_ref_dma_ack(p:*mut c_void,id:u32) { let t=model(p).transaction_mut(id);assert!(t.granted&&!t.dma_ack);t.dma_ack=true; }
#[no_mangle]
pub unsafe extern "C" fn interlock_ref_complete(p:*mut c_void,id:u32) {
    let m=model(p);let t=m.transaction(id);assert!(t.sealed);
    assert!(t.ranges.is_empty()||(t.granted&&t.dma_ack&&(!t.ranges[0].write||t.post.iter().all(|b|*b))),"premature completion");
    m.live.retain(|t|t.id!=id);
}
