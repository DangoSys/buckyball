package ack_pkg;
  import "DPI-C" function chandle ack_ref_create();
  import "DPI-C" function void ack_ref_destroy(input chandle model);
  import "DPI-C" function void ack_ref_write(
    input chandle model,
    input longint unsigned addr,
    lo,
    hi,
    input int unsigned mask
  );
  import "DPI-C" function int unsigned ack_ref_check(
    input chandle model,
    input longint unsigned addr,
    lo,
    hi
  );
endpackage
