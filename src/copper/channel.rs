//! # Copper Channels: Message Passing Infrastructure
//!
//! Copper's purpose is transmission, not storage.
//! Channels are the wires that connect regions.
//!
//! ## Channel Types
//!
//! - Point-to-point: One source, one destination
//! - Broadcast: One source, many destinations
//! - Gather: Many sources, one destination (aggregation)
//! - Bus: Many-to-many shared medium
//!
//! ## Unlike Iron
//!
//! Iron lattice settles into stable configurations.
//! Copper channels flow continuously - nothing stays still.

use super::lattice::FCCLattice;
use std::collections::VecDeque;

/// A message that flows through copper
#[derive(Clone, Debug)]
pub struct Message {
    /// The payload (could be concept signature, signal pattern, etc.)
    pub payload: Vec<f32>,
    /// Source identifier
    pub source: usize,
    /// Destination identifier (0 = broadcast)
    pub destination: usize,
    /// Priority (higher = faster routing)
    pub priority: u8,
    /// Time-to-live (decrements each hop)
    pub ttl: u8,
}

impl Message {
    pub fn new(payload: Vec<f32>, source: usize, destination: usize) -> Self {
        Self {
            payload,
            source,
            destination,
            priority: 5,
            ttl: 16,
        }
    }
    
    pub fn broadcast(payload: Vec<f32>, source: usize) -> Self {
        Self {
            payload,
            source,
            destination: 0,  // 0 = broadcast
            priority: 5,
            ttl: 8,  // Shorter TTL for broadcasts
        }
    }
}

/// A channel endpoint
#[derive(Clone, Debug)]
pub struct Endpoint {
    /// Unique identifier
    pub id: usize,
    /// Position in lattice (if spatially mapped)
    pub position: Option<(usize, usize, usize)>,
    /// Name/label
    pub name: String,
    /// Input buffer
    pub inbox: VecDeque<Message>,
    /// Output buffer
    pub outbox: VecDeque<Message>,
}

impl Endpoint {
    pub fn new(id: usize, name: &str) -> Self {
        Self {
            id,
            position: None,
            name: name.to_string(),
            inbox: VecDeque::new(),
            outbox: VecDeque::new(),
        }
    }
    
    pub fn at_position(mut self, x: usize, y: usize, z: usize) -> Self {
        self.position = Some((x, y, z));
        self
    }
    
    /// Send a message
    pub fn send(&mut self, payload: Vec<f32>, destination: usize) {
        let msg = Message::new(payload, self.id, destination);
        self.outbox.push_back(msg);
    }
    
    /// Broadcast to all
    pub fn broadcast(&mut self, payload: Vec<f32>) {
        let msg = Message::broadcast(payload, self.id);
        self.outbox.push_back(msg);
    }
    
    /// Receive next message
    pub fn receive(&mut self) -> Option<Message> {
        self.inbox.pop_front()
    }
    
    /// Peek at next message without removing
    pub fn peek(&self) -> Option<&Message> {
        self.inbox.front()
    }
    
    /// Number of pending messages
    pub fn pending(&self) -> usize {
        self.inbox.len()
    }
}

/// The copper channel network
pub struct ChannelNetwork {
    /// FCC lattice for signal routing
    pub lattice: FCCLattice,
    /// Registered endpoints
    pub endpoints: Vec<Endpoint>,
    /// Messages in transit
    pub in_transit: Vec<(Message, usize)>,  // (message, current_position)
}

impl ChannelNetwork {
    pub fn new(size: usize) -> Self {
        Self {
            lattice: FCCLattice::new(size, size, size),
            endpoints: Vec::new(),
            in_transit: Vec::new(),
        }
    }
    
    /// Register a new endpoint
    pub fn register(&mut self, name: &str) -> usize {
        let id = self.endpoints.len() + 1;  // IDs start at 1
        let endpoint = Endpoint::new(id, name);
        self.endpoints.push(endpoint);
        id
    }
    
    /// Register endpoint at specific position
    pub fn register_at(&mut self, name: &str, x: usize, y: usize, z: usize) -> usize {
        let id = self.endpoints.len() + 1;
        let endpoint = Endpoint::new(id, name).at_position(x, y, z);
        self.endpoints.push(endpoint);
        id
    }
    
    /// Get endpoint by ID
    pub fn endpoint(&self, id: usize) -> Option<&Endpoint> {
        self.endpoints.iter().find(|e| e.id == id)
    }
    
    /// Get mutable endpoint by ID
    pub fn endpoint_mut(&mut self, id: usize) -> Option<&mut Endpoint> {
        self.endpoints.iter_mut().find(|e| e.id == id)
    }
    
    /// Route messages for one step
    pub fn route(&mut self) {
        // Collect outgoing messages from all endpoints
        let mut new_transit: Vec<(Message, usize)> = Vec::new();
        
        for endpoint in &mut self.endpoints {
            while let Some(msg) = endpoint.outbox.pop_front() {
                // Find starting position
                let start_pos = if let Some((x, y, z)) = endpoint.position {
                    x + y * self.lattice.size_x + z * self.lattice.size_x * self.lattice.size_y
                } else {
                    // Default to center
                    let mid = self.lattice.size_x / 2;
                    mid + mid * self.lattice.size_x + mid * self.lattice.size_x * self.lattice.size_y
                };
                new_transit.push((msg, start_pos));
            }
        }
        
        // Take existing transit messages, avoiding borrow issues
        let old_transit: Vec<(Message, usize)> = std::mem::take(&mut self.in_transit);
        
        // Combine all messages
        let all_messages: Vec<(Message, usize)> = old_transit.into_iter()
            .chain(new_transit.into_iter())
            .collect();
        
        // Move messages toward destinations
        let mut delivered: Vec<(Message, usize)> = Vec::new();
        let mut still_transit: Vec<(Message, usize)> = Vec::new();
        
        for (mut msg, pos) in all_messages {
            // Decrement TTL
            if msg.ttl == 0 {
                continue;  // Drop expired messages
            }
            msg.ttl -= 1;
            
            // Check if at destination
            if msg.destination != 0 {
                // Point-to-point: find destination endpoint position
                let dest_info: Option<Option<(usize, usize, usize)>> = self.endpoints
                    .iter()
                    .find(|e| e.id == msg.destination)
                    .map(|e| e.position);
                    
                if let Some(dest_pos_opt) = dest_info {
                    if let Some((dx, dy, dz)) = dest_pos_opt {
                        let dest_pos = dx + dy * self.lattice.size_x + dz * self.lattice.size_x * self.lattice.size_y;
                        if pos == dest_pos {
                            let dest_id = msg.destination;
                            delivered.push((msg, dest_id));
                            continue;
                        }
                        // Move toward destination (gradient routing)
                        let new_pos = self.route_toward(pos, dest_pos);
                        still_transit.push((msg, new_pos));
                    } else {
                        // No position - deliver immediately
                        let dest_id = msg.destination;
                        delivered.push((msg, dest_id));
                    }
                }
            } else {
                // Broadcast: deliver to all endpoints we're near
                let source_id = msg.source;
                // Collect endpoint info first to avoid borrow issues
                let ep_info: Vec<(usize, Option<(usize, usize, usize)>)> = self.endpoints
                    .iter()
                    .map(|e| (e.id, e.position))
                    .collect();
                    
                for (ep_id, ep_pos) in &ep_info {
                    if *ep_id != source_id {
                        if let Some((ex, ey, ez)) = ep_pos {
                            let (px, py, pz) = self.lattice.coords(pos);
                            let dist = ((px as i32 - *ex as i32).abs() + 
                                       (py as i32 - *ey as i32).abs() +
                                       (pz as i32 - *ez as i32).abs()) as usize;
                            if dist <= 2 {
                                delivered.push((msg.clone(), *ep_id));
                            }
                        } else {
                            // No position - always deliver broadcasts
                            delivered.push((msg.clone(), *ep_id));
                        }
                    }
                }
                // Keep propagating if TTL > 0
                if msg.ttl > 0 {
                    let new_pos = self.diffuse(pos);
                    still_transit.push((msg, new_pos));
                }
            }
        }
        
        self.in_transit = still_transit;
        
        // Deliver messages
        for (msg, dest_id) in delivered {
            if let Some(ep) = self.endpoint_mut(dest_id) {
                ep.inbox.push_back(msg);
            }
        }
    }
    
    /// Route toward destination using gradient following
    fn route_toward(&self, current: usize, destination: usize) -> usize {
        let (dx, dy, dz) = self.lattice.coords(destination);
        
        // Find neighbor closest to destination
        let node = &self.lattice.nodes[current];
        let mut best = current;
        let mut best_dist = i32::MAX;
        
        for &neighbor in &node.neighbors {
            let (nx, ny, nz) = self.lattice.coords(neighbor);
            let dist = (nx as i32 - dx as i32).abs() +
                      (ny as i32 - dy as i32).abs() +
                      (nz as i32 - dz as i32).abs();
            if dist < best_dist {
                best_dist = dist;
                best = neighbor;
            }
        }
        
        best
    }
    
    /// Diffuse to random neighbor (for broadcasts)
    fn diffuse(&self, current: usize) -> usize {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let node = &self.lattice.nodes[current];
        let idx = rng.gen_range(0..12);
        node.neighbors[idx]
    }
    
    /// Run multiple routing steps
    pub fn step(&mut self, steps: usize) {
        for _ in 0..steps {
            self.route();
        }
    }
    
    /// Messages currently in flight
    pub fn in_flight(&self) -> usize {
        self.in_transit.len()
    }
}

/// Demo copper channels
pub fn demo_copper_channels() {
    println!("=== COPPER CHANNELS: Message Passing ===\n");
    
    let mut network = ChannelNetwork::new(8);
    
    // Register endpoints
    let alice = network.register_at("alice", 0, 0, 0);
    let bob = network.register_at("bob", 7, 7, 7);
    let charlie = network.register_at("charlie", 4, 4, 4);
    
    println!("Registered endpoints:");
    println!("  Alice (id={}) at (0,0,0)", alice);
    println!("  Bob (id={}) at (7,7,7)", bob);
    println!("  Charlie (id={}) at (4,4,4)", charlie);
    println!();
    
    // Alice sends to Bob
    if let Some(ep) = network.endpoint_mut(alice) {
        ep.send(vec![1.0, 2.0, 3.0], bob);
        println!("Alice sends [1,2,3] to Bob");
    }
    
    // Charlie broadcasts
    if let Some(ep) = network.endpoint_mut(charlie) {
        ep.broadcast(vec![9.0, 9.0, 9.0]);
        println!("Charlie broadcasts [9,9,9]");
    }
    
    println!("\nRouting messages...");
    
    // Route until delivered
    for step in 0..20 {
        network.route();
        
        // Check for deliveries
        if let Some(ep) = network.endpoint_mut(bob) {
            while let Some(msg) = ep.receive() {
                println!("  Step {}: Bob received {:?} from {}", 
                        step, msg.payload, msg.source);
            }
        }
        if let Some(ep) = network.endpoint_mut(alice) {
            while let Some(msg) = ep.receive() {
                println!("  Step {}: Alice received {:?} from {}", 
                        step, msg.payload, msg.source);
            }
        }
        
        if network.in_flight() == 0 {
            println!("\nAll messages delivered by step {}", step);
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_point_to_point() {
        let mut network = ChannelNetwork::new(4);
        let a = network.register_at("a", 0, 0, 0);
        let b = network.register_at("b", 3, 3, 3);
        
        network.endpoint_mut(a).unwrap().send(vec![42.0], b);
        
        // Route until delivered
        for _ in 0..20 {
            network.route();
        }
        
        let msg = network.endpoint_mut(b).unwrap().receive();
        assert!(msg.is_some());
        assert_eq!(msg.unwrap().payload, vec![42.0]);
    }
}
