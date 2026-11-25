class coverage extends uvm_subscriber #(command);
  `uvm_component_utils(coverage)
  command observed;
  covergroup behavior;
    option.per_instance = 1;
    direction: coverpoint observed.write;
    start_bank: coverpoint (observed.word_addr % `BANK_COUNT) {
      bins banks[] = {[0 : `BANK_COUNT - 1]};
    }
    length: coverpoint observed.beats {
      bins single = {1}; bins partial = {[2 : `BANK_WORDS - 1]}; bins full = {`BANK_WORDS};
    }
    crossed_row: coverpoint ((observed.word_addr % `BANK_COUNT) + observed.beats > `BANK_COUNT);
    stalled: coverpoint (observed.completion_stall > 1);
    direction_bank: cross direction, start_bank;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    behavior = new();
  endfunction
  function void write(command t);
    observed = t;
    behavior.sample();
  endfunction
endclass
