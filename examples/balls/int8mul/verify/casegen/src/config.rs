pub fn bank_entries() -> u32 {
    let path = std::env::var("BB_VERIFY_CONFIG").expect("BB_VERIFY_CONFIG is required");
    let text = std::fs::read_to_string(path).expect("failed to read BB_VERIFY_CONFIG");
    text.lines()
        .find_map(|line| line.strip_prefix("bank_entries="))
        .unwrap_or_else(|| panic!("bank_entries missing from BB_VERIFY_CONFIG"))
        .parse()
        .expect("invalid bank_entries in BB_VERIFY_CONFIG")
}
