// Reset while the group is held and a task is live clears the group, ownership and
// completion; both contexts report idle afterwards.
task automatic control_reset();
  transact(5, 0, 0, 0);
  descriptor(0);
  transact(2, 0, 0, 0);
  @(negedge clock);
  reset = 1;
  owned = '{0, 0};
  done  = '{0, 0};
  repeat (2) @(negedge clock);
  reset = 0;
  if (in_use || available !== 2'b11 || cancel || start_valid)
    `uvm_fatal("RESET", "reset left the group or a task behind")
  transact(0, 0, 1, 0, 1);
  transact(0, 1, 1, 0, 1);
endtask
