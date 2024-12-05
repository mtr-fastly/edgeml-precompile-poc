use rand::{distributions::Alphanumeric, Rng};
use log::{info, Level, log};

pub fn emit_log(context: &str, session: &str, msg: &str) {
    emit_log_at_level(Level::Info, context, session, msg)
}

pub fn emit_log_at_level(level:Level, context: &str, session: &str, msg: &str) {
    let log_id: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(7)
        .map(char::from)
        .collect();
    println!("{{\"logID\":\"edgeml{}\",\"source\":\"{}\",\"context\":\"{}\",\"session\":\"{}\",\"msg\":\"{}\"}}", std::env::var("FASTLY_HOSTNAME").unwrap_or_default(), log_id, context, session, msg);
}

