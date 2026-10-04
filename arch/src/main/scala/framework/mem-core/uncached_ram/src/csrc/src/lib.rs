use std::{collections::BTreeMap, ffi::c_void};
struct Memory { bytes:BTreeMap<u64,u8>, reservations:[Option<(u64,u32)>;2] }
#[no_mangle] pub extern "C" fn ram_ref_create()->*mut c_void { Box::into_raw(Box::new(Memory{bytes:BTreeMap::new(),reservations:[None;2]})).cast() }
#[no_mangle] pub unsafe extern "C" fn ram_ref_destroy(p:*mut c_void){drop(Box::from_raw(p.cast::<Memory>()));}
#[no_mangle] pub unsafe extern "C" fn ram_ref_reset(p:*mut c_void){(&mut *p.cast::<Memory>()).reservations=[None;2];}
#[no_mangle] pub unsafe extern "C" fn ram_ref_program(p:*mut c_void,addr:u64,data:*const u32){
 let m=&mut *p.cast::<Memory>();for i in 0..64{m.bytes.insert(addr+i,((*data.add(i as usize/4)>>((i%4)*8))&255)as u8);}
}
#[no_mangle] pub unsafe extern "C" fn ram_ref_read(p:*mut c_void,addr:u64,data:*mut u32){
 let m=&*p.cast::<Memory>();for i in 0..16{*data.add(i)=0;}for i in 0..64{*data.add(i as usize/4)|=u32::from(*m.bytes.get(&(addr+i)).expect("uninitialized reference byte"))<<((i%4)*8);}
}
fn invalidate(m:&mut Memory,addr:u64){for r in &mut m.reservations{if r.is_some_and(|(a,_)|a/64==addr/64){*r=None;}}}
#[no_mangle] pub unsafe extern "C" fn ram_ref_line_write(p:*mut c_void,addr:u64,data:*const u32,mask:*const u32,failed:u32){
 let m=&mut *p.cast::<Memory>();let mask=u64::from(*mask)|(u64::from(*mask.add(1))<<32);
 if mask!=0{invalidate(m,addr);}if failed==0{for i in 0..64{if mask&(1<<i)!=0{*m.bytes.get_mut(&(addr+i)).expect("uninitialized write byte")=((*data.add(i as usize/4)>>((i%4)*8))&255)as u8;}}}
}
#[no_mangle] pub unsafe extern "C" fn ram_ref_cpu(p:*mut c_void,cpu:u32,addr:u64,size:u32,write:u32,op:u32,operand:u64,fault:u32,result:*mut u64)->u32{
 let m=&mut *p.cast::<Memory>();let cpu=cpu as usize;assert!(cpu<2&&size<=3&&op<=11);
 let width=1u64<<size;let reserve=m.reservations[cpu];if op==10||op==11{m.reservations[cpu]=None;}
 *result=0;
 if addr<0x80000000||addr.checked_add(width).is_none_or(|end|end>0x81000000)||addr%width!=0{return 1;}
 let sc=op==11;let success=reserve==Some((addr,size));
 if sc&&!success{*result=1;return 0;}
 let writing=write!=0||(op!=0&&op!=10);if writing{invalidate(m,addr);}
 if (write!=0||sc)&&fault&2!=0{return 1;}if write==0&&!sc&&fault&1!=0{return 1;}
 let mut old=0u64;for i in 0..width{old|=u64::from(*m.bytes.get(&(addr+i)).expect("uninitialized CPU byte"))<<(8*i);}
 let word=size==2;let rhs=if word{operand&0xffffffff}else{operand};
 let signed=|x:u64|if word{x as u32 as i32 as i64}else{x as i64};
 let new=if write!=0||sc{operand}else{match op{0|10=>old,1=>rhs,2=>old.wrapping_add(rhs),3=>old^rhs,4=>old&rhs,5=>old|rhs,6=>if signed(old)<signed(rhs){old}else{rhs},7=>if signed(old)>signed(rhs){old}else{rhs},8=>old.min(rhs),9=>old.max(rhs),_=>unreachable!()}};
 if op!=0&&op!=10&&!sc&&fault&2!=0{return 1;}
 if writing{for i in 0..width{*m.bytes.get_mut(&(addr+i)).unwrap()=(new>>(8*i))as u8;}}
 if op==10{m.reservations[cpu]=Some((addr,size));}
 if write==0&&!sc{*result=if op!=0&&word{old as u32 as i32 as i64 as u64}else{old};}
 0
}
