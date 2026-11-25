use std::cell::RefCell;
use crate::reference;
struct Case { rows: usize, k: usize, alias: bool, reserved: bool, reject: bool, words: Vec<Vec<u32>>, lookups: Vec<u32> }
thread_local! { static CASE: RefCell<Option<Case>> = const { RefCell::new(None) }; }
#[no_mangle]
pub extern "C" fn intsoftmax_case_load(index: u32) {
    assert!(index<20);
    let (rows,k) = if index<11 {
        [(1,1),(2,7),(3,32),(8,128),(32,128),(1024,1),(2,127),(4,65),(3,4),(1,3),(2,6)][index as usize]
    } else { (1,if index==17 || index==19 {2}else{1}) };
    let mut input=vec![0x7fc12345;4096];
    let mut table: Vec<u32> = (0..4096).map(|i| ((-i as f64/256.0).exp()*32768.0).round_ties_even() as u32).collect();
    let mut seed=0x12345678u32;
    for row in 0..rows { for col in 0..k {
        seed=seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let bits=match index {
            1 => [0x3b000000,0x3b000001,0xbb000000,0xbb000001,0x3bc00000,0xbbc00000,0][col%7],
            2 => if col>row { f32::NEG_INFINITY.to_bits() } else { ((row+col) as f32/8.0).to_bits() },
            5|6 => if seed&0x7f800000==0x7f800000 { seed&0xff800000 } else { seed },
            7 => [0x7f800000,0xff800000,0x7f7fffff,0xff7fffff,0x45800000,0xc5800000][col%6],
            8 => [0,0x80000000,1,0x007fffff][col%4],
            _ => ((seed%4097) as f32/256.0-8.0+row as f32/64.0).to_bits(),
        };
        input[row*k.div_ceil(4)*4+col]=bits;
    }}
    if index==11 { input[0]=0x7fc12345; }
    if index==16 { input[0]=0; table[0]=0; }
    if index==17 { input[0]=0; input[1]=(-4096.0f32).to_bits(); table[4095]=32769; }
    if index==19 { input[0]=0; input[1]=0; table[4095]=u32::MAX; }
    let reject=(11..=18).contains(&index);
    let (output,lookups)=if reject {
        (vec![0x5a5a5a5a;4096],if index==17 {vec![0,4095]}else{vec![0]})
    } else { reference::expected(&input,&table,rows,k) };
    let rows=if index==14 {33}else{rows};
    let k=match index {12=>0,13=>129,14=>128,_=>k};
    CASE.with(|c| *c.borrow_mut()=Some(Case{rows,k,alias:index==15,reserved:index==18,reject,words:vec![input,table,output],lookups}));
}
#[no_mangle]
pub extern "C" fn intsoftmax_case_field(field: u32) -> u32 {
    CASE.with(|c| { let c=c.borrow();let c=c.as_ref().unwrap();[c.rows,c.k,c.alias as usize,c.reserved as usize,c.reject as usize][field as usize] as u32 })
}
#[no_mangle]
pub extern "C" fn intsoftmax_case_word(role: u32, row: u32, high: u32) -> u64 {
    CASE.with(|c| { let c=c.borrow();let v=&c.as_ref().unwrap().words[role as usize];let at=row as usize*4+high as usize*2;v[at] as u64|((v[at+1] as u64)<<32) })
}
#[no_mangle]
pub extern "C" fn intsoftmax_case_lookup(position: u32) -> u32 {
    CASE.with(|c| c.borrow().as_ref().unwrap().lookups[position as usize])
}
