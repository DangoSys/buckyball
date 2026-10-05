use crate::mmu;
use std::{collections::{HashMap, VecDeque}, ffi::c_void};

#[derive(Clone, Copy)]
struct Segment { va:u64, pa:u64, bytes:u32, error:u32 }
#[derive(Clone, Copy)]
struct Auth { pa:u64, bytes:u32, write:u32, pte:u32, privilege:u32, allow:u32 }
struct Plan { output:Vec<Segment>, auth:VecDeque<Auth>, ptes:VecDeque<u64>, write:u32 }
struct Model {
    mmu:*mut c_void, address_bits:u32, beat:u64,
    plans:HashMap<u32,Plan>, deny_pte:Option<u64>, deny_range:Option<(u64,u64)>,
}
#[no_mangle]
pub extern "C" fn preflight_ref_create(address_bits:u32, beat:u32)->*mut c_void {
    Box::into_raw(Box::new(Model{mmu:mmu::mmu_ref_create(address_bits),address_bits,beat:beat as u64,
        plans:HashMap::new(),deny_pte:None,deny_range:None})).cast()
}
#[no_mangle]
pub unsafe extern "C" fn preflight_ref_destroy(ptr:*mut c_void) {
    let model=Box::from_raw(ptr.cast::<Model>());mmu::mmu_ref_destroy(model.mmu);
}
#[no_mangle]
pub unsafe extern "C" fn preflight_ref_program(ptr:*mut c_void,pa:u64,value:u64,error:u32) {
    mmu::mmu_ref_program((*ptr.cast::<Model>()).mmu,pa,value,error);
}
#[no_mangle]
pub unsafe extern "C" fn preflight_ref_policy(ptr:*mut c_void,pte_valid:u32,pte:u64,range_valid:u32,first:u64,last:u64) {
    let m=&mut *ptr.cast::<Model>();m.deny_pte=(pte_valid!=0).then_some(pte);m.deny_range=(range_valid!=0).then_some((first,last));
}
#[no_mangle]
pub unsafe extern "C" fn preflight_ref_submit(ptr:*mut c_void,id:u32,base:u64,rows:u32,cols:u32,span:u32,
    col_stride:u32,row_stride:u32,write:u32,mode:u32,root:u64,privilege:u32,sum:u32,mxr:u32)->u32 {
    let m=&mut *ptr.cast::<Model>();assert!(!m.plans.contains_key(&id));
    let mut plan=Plan{output:Vec::new(),auth:VecDeque::new(),ptes:VecDeque::new(),write};
    let mut fault=None;
    let dense=(cols==1||col_stride==span)&&(rows==1||row_stride as u128==cols as u128*span as u128);
    if rows==0||cols==0||span==0||(rows>1&&row_stride==0)||(cols>1&&col_stride==0) {fault=Some((1,base));}
    else if !matches!(mode,0|8)||!matches!(privilege,0|1|3) {fault=Some((6,base));}
    else {
        let end=base as u128+(rows-1) as u128*row_stride as u128+(cols-1) as u128*col_stride as u128+span as u128-1;
        if end>u64::MAX as u128||(dense&&rows as u128*cols as u128*span as u128>u32::MAX as u128) {fault=Some((2,base));}
    }
    // Only the pure architectural walker is reused. Shape iteration and queues
    // here are sequential, independent of the RTL's slots and state machine.
    if fault.is_none() {
        'shape: for y in 0..if dense {1}else{rows} {
            for x in 0..if dense {1}else{cols} {
                let start=base as u128+y as u128*row_stride as u128+x as u128*col_stride as u128;
                let length=if dense {rows as u128*cols as u128*span as u128}else{span as u128};
                let first=if write!=0 {start}else{start & !(m.beat as u128-1)};
                let end=if write!=0 {start+length}else{(start+length+m.beat as u128-1)&!(m.beat as u128-1)};
                if end-first>u32::MAX as u128 {fault=Some((2,start as u64));break 'shape;}
                let mut current=first;
                while current<end {
                    let va=current as u64;
                    let bytes=(end-current).min(4096-(current&4095)) as u32;
                    let mut saved=None;
                    if let Some(address)=m.deny_pte {
                        let(mut value,mut error)=(0,0);
                        if mmu::mmu_ref_read(m.mmu,address,&mut value,&mut error)==1 {
                            saved=Some((address,value,error));mmu::mmu_ref_program(m.mmu,address,value,1);
                        }
                    }
                    let(mut pa,mut pf,mut af,mut level)=(0,0,0,0);
                    let missing=mmu::mmu_ref_translate(m.mmu,va,mode,root,privilege,write,0,sum,mxr,&mut pa,&mut pf,&mut af,&mut level);
                    if let Some((a,v,e))=saved {mmu::mmu_ref_program(m.mmu,a,v,e);}
                    if missing!=0 {return 1;}
                    let mut address=0;
                    while mmu::mmu_ref_peek(m.mmu,&mut address)!=0 {
                        let allow=(m.deny_pte!=Some(address)) as u32;
                        plan.auth.push_back(Auth{pa:address,bytes:8,write:0,pte:1,privilege:1,allow});
                        if allow!=0 {plan.ptes.push_back(address);}
                        assert_eq!(mmu::mmu_ref_consume(m.mmu,address),1);
                    }
                    if pf!=0||af!=0 {fault=Some((if pf!=0 {3}else{4},va));break 'shape;}
                    if pa as u128+bytes as u128>(1u128<<m.address_bits) {fault=Some((4,va));break 'shape;}
                    let permitted=|address:u64,length:u32| !m.deny_range.is_some_and(|(lo,hi)|address<=hi&&address+length as u64-1>=lo);
                    let union=plan.output.last().filter(|p|p.va>>12==va>>12&&va>p.va+p.bytes as u64&&p.pa+(va-p.va)==pa)
                        .map(|p|(p.pa,(va+bytes as u64-p.va) as u32));
                    let union_allowed=if let Some((address,length))=union {
                        let allow=permitted(address,length);
                        plan.auth.push_back(Auth{pa:address,bytes:length,write,pte:0,privilege,allow:allow as u32});
                        allow
                    }else{false};
                    let allow=union_allowed||permitted(pa,bytes);
                    if !union_allowed {plan.auth.push_back(Auth{pa,bytes,write,pte:0,privilege,allow:allow as u32});}
                    if !allow {fault=Some((4,va));break 'shape;}
                    // Retain exactly the union permitted above, or the separately permitted spans.
                    let merge=union_allowed||plan.output.last().is_some_and(|p| p.va>>12==va>>12&&va>=p.va&&va<=p.va+p.bytes as u64);
                    if merge {
                        let p=plan.output.last_mut().unwrap();
                        assert_eq!(p.pa+(va-p.va),pa);
                        let end=va+bytes as u64;
                        if end>p.va+p.bytes as u64 {p.bytes=(end-p.va) as u32;}
                    } else if plan.output.len()==8 {fault=Some((5,va));break 'shape;}
                    else {plan.output.push(Segment{va,pa,bytes,error:0});}
                    current+=bytes as u128;
                }
            }
        }
    }
    if let Some((error,va))=fault {plan.output=vec![Segment{va,pa:0,bytes:0,error}];}
    m.plans.insert(id,plan);0
}
#[no_mangle]
pub unsafe extern "C" fn preflight_ref_count(ptr:*mut c_void,id:u32)->u32 {let m=&*ptr.cast::<Model>();m.plans[&id].output.len() as u32}
#[no_mangle]
pub unsafe extern "C" fn preflight_ref_expected(ptr:*mut c_void,id:u32,index:u32,va:*mut u64,pa:*mut u64,bytes:*mut u32,write:*mut u32,last:*mut u32,error:*mut u32) {
    let m=&*ptr.cast::<Model>();let plan=&m.plans[&id];let s=plan.output[index as usize];
    *va=s.va;*pa=s.pa;*bytes=s.bytes;*write=plan.write;*last=(index as usize+1==plan.output.len()) as u32;*error=s.error;
}
#[no_mangle]
pub unsafe extern "C" fn preflight_ref_authorize(ptr:*mut c_void,id:u32,pa:u64,bytes:u32,write:u32,pte:u32,privilege:u32,allow:*mut u32)->u32 {
    let m=&mut *ptr.cast::<Model>();let Some(plan)=m.plans.get_mut(&id) else{return 0};
    let Some(a)=plan.auth.pop_front() else{return 0};
    *allow=a.allow;(a.pa==pa&&a.bytes==bytes&&a.write==write&&a.pte==pte&&a.privilege==privilege) as u32
}
#[no_mangle]
pub unsafe extern "C" fn preflight_ref_pte(ptr:*mut c_void,id:u32,address:u64,value:*mut u64,error:*mut u32)->u32 {
    let m=&mut *ptr.cast::<Model>();let Some(plan)=m.plans.get_mut(&id) else{return 0};
    if plan.ptes.pop_front()!=Some(address) {return 0;}
    mmu::mmu_ref_read(m.mmu,address,value,error)
}
#[no_mangle]
pub unsafe extern "C" fn preflight_ref_pending(ptr:*mut c_void,id:u32)->u32 {
    let m=&*ptr.cast::<Model>();let plan=&m.plans[&id];(plan.auth.len()+plan.ptes.len()) as u32
}
#[no_mangle]
pub unsafe extern "C" fn preflight_ref_retire(ptr:*mut c_void,id:u32) {(*ptr.cast::<Model>()).plans.remove(&id).expect("unknown preparation");}
#[no_mangle]
pub unsafe extern "C" fn preflight_ref_reset(ptr:*mut c_void) {(*ptr.cast::<Model>()).plans.clear();}
