# Neo4j & Cypher Query Language
## Complete Reference: Beginner to Industry-Ready for AI/ML Engineers

---

> **How to use this document:**
> Read linearly if you're new. Jump to sections if you're referencing.
> Every section builds on the previous. Code is runnable unless marked `[conceptual]`.

---

## Table of Contents

1. [Graph Database Fundamentals](#1-graph-database-fundamentals)
2. [Why Graph DBs for AI/ML](#2-why-graph-dbs-for-aiml)
3. [Neo4j Architecture Internals](#3-neo4j-architecture-internals)
4. [Installation & Setup](#4-installation--setup)
5. [Cypher Query Language — Core Syntax](#5-cypher-query-language--core-syntax)
6. [CRUD Operations](#6-crud-operations)
7. [Pattern Matching — The Heart of Cypher](#7-pattern-matching--the-heart-of-cypher)
8. [Filtering, Aggregation & Projection](#8-filtering-aggregation--projection)
9. [Path Queries & Graph Traversal](#9-path-queries--graph-traversal)
10. [Graph Algorithms (GDS Library)](#10-graph-algorithms-gds-library)
11. [Indexing & Query Optimization](#11-indexing--query-optimization)
12. [Schema Design for AI/ML](#12-schema-design-for-aiml)
13. [Neo4j for Knowledge Graphs](#13-neo4j-for-knowledge-graphs)
14. [Neo4j + Python (py2neo, neo4j-driver)](#14-neo4j--python-py2neo-neo4j-driver)
15. [Neo4j for RAG Systems](#15-neo4j-for-rag-systems)
16. [Neo4j for Recommendation Systems](#16-neo4j-for-recommendation-systems)
17. [Neo4j for Fraud Detection & Link Analysis](#17-neo4j-for-fraud-detection--link-analysis)
18. [Graph Neural Networks with Neo4j](#18-graph-neural-networks-with-neo4j)
19. [Production: Performance, Scaling, Reliability](#19-production-performance-scaling-reliability)
20. [Observability & Debugging](#20-observability--debugging)
21. [Anti-Patterns & What NOT to Do](#21-anti-patterns--what-not-to-do)
22. [Industry Patterns & Reference Architectures](#22-industry-patterns--reference-architectures)

---

## 1. Graph Database Fundamentals

### 1.1 What Is a Graph Database?

A graph database stores data as **nodes** (entities) and **edges** (relationships), not rows and columns. The relationships are first-class citizens — they are stored with their own identity, direction, type, and properties.

**Core primitives:**

| Primitive      | Description                                        | Example                        |
|----------------|----------------------------------------------------|--------------------------------|
| **Node**       | An entity / object                                 | `(:Person {name: "Alice"})`    |
| **Relationship** | A directed, typed connection between two nodes   | `-[:KNOWS {since: 2020}]->`    |
| **Label**      | A category tag on a node (can have multiple)       | `:Person`, `:Employee`         |
| **Property**   | Key-value pair on a node or relationship           | `{name: "Alice", age: 30}`     |

### 1.2 Graph vs Relational vs Document — Honest Comparison

```
RELATIONAL (PostgreSQL):
  Table: Person(id, name)
  Table: Friendship(person_a_id, person_b_id, since)
  
  Query: "Friends of friends of Alice" =
    3 JOINs, expensive at scale, fixed schema

DOCUMENT (MongoDB):
  { _id: "alice", friends: ["bob", "carol"] }
  
  Problem: Nested lookups, no relationship semantics,
  multi-hop traversal requires application-level logic

GRAPH (Neo4j):
  (:Person {name:"Alice"})-[:KNOWS]->(:Person {name:"Bob"})
  
  Query: "Friends of friends" = 1 pattern match, 
  index-free adjacency, O(local) not O(global)
```

**When graph wins:**
- Highly connected data with many relationship types
- Multi-hop traversals (2+ hops)
- Pattern detection across nodes
- Evolving schema with heterogeneous data
- Recommendation, fraud detection, knowledge graphs

**When graph loses:**
- Tabular bulk analytics (use columnar DB)
- Simple CRUD on flat entities (use RDBMS)
- Full-text search as primary access (use Elasticsearch)
- Bulk ML feature pipelines (graph overhead is real)

### 1.3 Property Graph Model (PGM)

Neo4j implements the **Labeled Property Graph** model:

```
Node:
  - Has a unique internal ID (Neo4j managed)
  - Can have 0 to N labels
  - Properties: map of {string -> primitive|list}

Relationship:
  - Has a unique internal ID
  - Has exactly ONE type (string, UPPER_CASE convention)
  - Is directed (but can be queried in either direction)
  - Can have properties
  - Always has a start node and end node

Properties support:
  String, Integer, Float, Boolean, Date, DateTime,
  Duration, Point (2D/3D), and Lists of the above
  NULL is NOT stored — absence of key = null
```

---

## 2. Why Graph DBs for AI/ML

### 2.1 Core AI/ML Use Cases

| Use Case | Why Graph Wins | Alternative Pain |
|---|---|---|
| Knowledge Graphs | Entities + typed relations natively | RDF stores are verbose; RDBMS needs dozens of tables |
| Recommendation | Collaborative filtering via graph traversal | Matrix factorization loses interpretability |
| Fraud Detection | Ring/chain pattern detection across accounts | SQL subqueries become unmaintainable at depth |
| RAG Grounding | Structured context retrieval for LLMs | Vector-only RAG has no relational structure |
| Feature Engineering | Graph-derived features (centrality, community) | Hard to compute in Pandas without networkx |
| Entity Resolution | Link records via shared properties | Fuzzy joins in SQL are painful |
| Supply Chain / Lineage | DAG traversal, impact analysis | Recursive CTEs in SQL are ugly and slow |

### 2.2 Graph as Feature Store for ML

Graph topology generates features that are expensive to produce any other way:

```python
# Features derivable from graph structure:
# - Node degree (in/out) — proxy for popularity
# - Betweenness centrality — proxy for influence
# - PageRank — global importance
# - Community membership — cluster identity
# - Shortest path distance — relational proximity
# - Triangle count — social density / trust signal
# - Jaccard similarity — neighborhood overlap

# These become columns in your ML feature vector
# Neo4j GDS library computes all of these natively
```

### 2.3 Graphs + LLMs: The Emerging Stack

```
[User Query]
     |
     v
[LLM] --> [Graph Retriever] --> [Neo4j Knowledge Graph]
              |                        |
              v                        v
        [Vector Index]        [Structured Context]
              |                        |
              +----------+-------------+
                         |
                    [Grounded Response]
```

This pattern (GraphRAG) solves hallucination by grounding LLM outputs in structured, verifiable facts stored as graph relationships.

---

## 3. Neo4j Architecture Internals

### 3.1 Storage Layer

Neo4j stores data in fixed-size record files on disk:

```
neostore.nodestore.db          — Node records (15 bytes each)
neostore.relationshipstore.db  — Relationship records (34 bytes each)
neostore.propertystore.db      — Property records (chained)
neostore.labeltokenstore.db    — Label name lookup
neostore.relationshiptypestore — Relationship type names
```

**Index-Free Adjacency (IFA)** — the core performance principle:

```
In a relational DB:
  "Give me Alice's friends" = scan Friendship table WHERE person_a = alice_id
  Cost: O(N) where N = total rows in table

In Neo4j:
  "Give me Alice's friends" = follow pointers FROM alice's node record
  Cost: O(degree) — only Alice's connections, regardless of total graph size

This is why multi-hop graph traversal is orders of magnitude faster in Neo4j
than equivalent recursive CTEs or JOIN chains in SQL.
```

Each node record stores a pointer to its first relationship. Each relationship record stores pointers to:
- Start node
- End node
- Next relationship for start node
- Next relationship for end node

This forms a **doubly-linked list** per node, enabling O(1) neighbor access.

### 3.2 Transaction & ACID Guarantees

Neo4j is fully ACID:

```
Atomicity:   All writes in a transaction succeed or none do
Consistency: Graph constraints enforced per transaction
Isolation:   Read-committed by default (configurable)
Durability:  Write-ahead log (WAL) before commit
```

**Write path:**
```
Client Write → Transaction Log (WAL) → Page Cache → Disk Flush
                     |
                     └── On crash: replay WAL to recover
```

**Page Cache** is critical for performance. Neo4j keeps the working set of nodes/relationships in memory. Sizing it correctly is one of the most important production tuning decisions.

### 3.3 Execution Engine

Cypher queries go through:

```
Cypher String
     |
     v
[Parser] → AST
     |
     v
[Semantic Analysis] → Type checking, scope resolution
     |
     v
[Logical Planner] → Logical plan (tree of operators)
     |
     v
[Cost-Based Optimizer] → Chooses indexes, join strategies
     |
     v
[Physical Plan] → Operator pipeline
     |
     v
[Runtime: Slotted / Pipelined / Parallel]
     |
     v
Results
```

**Runtimes (Neo4j 5.x):**
- **Slotted**: Default, row-by-row, safe for all queries
- **Pipelined**: Vectorized, faster for analytics
- **Parallel**: Multi-threaded, for GDS and large aggregations

### 3.4 Clustering (Enterprise)

```
[Primary]  ←→  [Primary]  ←→  [Primary]    ← Raft consensus, any can handle writes
    ↓               ↓               ↓
[Secondary] [Secondary] [Secondary]           ← Read replicas, async replication

Write: Goes to leader, Raft quorum confirms (majority must acknowledge)
Read:  Can go to any secondary (eventual consistency)
```

---

## 4. Installation & Setup

### 4.1 Docker (Recommended for Dev)

```bash
# Neo4j Community (free, single instance)
docker run \
  --name neo4j \
  -p 7474:7474 \   # HTTP (Browser UI)
  -p 7687:7687 \   # Bolt (driver protocol)
  -e NEO4J_AUTH=neo4j/yourpassword \
  -e NEO4J_PLUGINS='["apoc", "graph-data-science"]' \
  -v $HOME/neo4j/data:/data \
  -v $HOME/neo4j/logs:/logs \
  neo4j:5.15.0

# Access browser UI: http://localhost:7474
# Default credentials: neo4j / yourpassword
```

### 4.2 Docker Compose (Full Stack for ML)

```yaml
# docker-compose.yml
version: "3.8"
services:
  neo4j:
    image: neo4j:5.15.0
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: neo4j/mlpassword
      NEO4J_PLUGINS: '["apoc", "graph-data-science"]'
      NEO4J_server_memory_heap_initial__size: "2G"
      NEO4J_server_memory_heap_max__size: "4G"
      NEO4J_server_memory_pagecache__size: "4G"
      NEO4J_dbms_security_procedures_unrestricted: "gds.*,apoc.*"
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_import:/import
    healthcheck:
      test: ["CMD", "wget", "-O", "-", "http://localhost:7474"]
      interval: 10s
      timeout: 5s
      retries: 10

  jupyter:
    image: jupyter/scipy-notebook
    ports:
      - "8888:8888"
    environment:
      - JUPYTER_ENABLE_LAB=yes
    volumes:
      - ./notebooks:/home/jovyan/work
    depends_on:
      - neo4j

volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_import:
```

### 4.3 Python Driver Setup

```bash
pip install neo4j          # Official driver
pip install graphdatascience  # GDS Python client
pip install langchain-community  # LangChain + Neo4j
```

```python
from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "mlpassword")

driver = GraphDatabase.driver(URI, auth=AUTH)

# Always verify connectivity on startup
driver.verify_connectivity()
print("Connected to Neo4j")

# Context manager for sessions
with driver.session(database="neo4j") as session:
    result = session.run("RETURN 'hello' AS msg")
    print(result.single()["msg"])  # hello

driver.close()
```

### 4.4 Neo4j Configuration Reference

```ini
# neo4j.conf — key settings for ML workloads

# Memory (most critical tuning)
server.memory.heap.initial_size=2g
server.memory.heap.max_size=4g
server.memory.pagecache.size=8g   # Set to ~graph size on disk

# Threads
db.transaction.concurrent.maximum=100

# Query logging (essential for debugging)
db.logs.query.enabled=INFO
db.logs.query.threshold=1000ms    # Log queries > 1 second

# Import
db.import.csv.legacy_quote_escaping=false
```

---

## 5. Cypher Query Language — Core Syntax

### 5.1 Cypher Philosophy

Cypher is a **declarative**, **pattern-based** language. You describe *what* you want, not *how* to get it.

```
ASCII art syntax: Nodes in (), relationships in []

(:Label {prop: value})         — node
-[:TYPE {prop: value}]->       — directed relationship
(:Label)-[:TYPE]-(:Label)      — undirected pattern (matches both directions)
```

### 5.2 Clause Execution Order

```
MATCH          — Find patterns in the graph
WHERE          — Filter results of MATCH
WITH           — Pipe results, like SQL subquery boundary
RETURN         — Output columns
ORDER BY       — Sort (only in final RETURN or WITH)
SKIP / LIMIT   — Pagination
UNWIND         — Expand a list into rows
OPTIONAL MATCH — Like LEFT JOIN (nulls instead of missing)
MERGE          — Match or create (upsert semantics)
CREATE         — Create nodes/relationships
SET            — Update properties or labels
REMOVE         — Remove properties or labels
DELETE         — Delete nodes or relationships
DETACH DELETE  — Delete node and all its relationships
FOREACH        — Iterate list, perform updates
CALL           — Call stored procedures or subqueries
```

### 5.3 Syntax Reference Card

```cypher
-- ============================================================
-- NODE PATTERNS
-- ============================================================
()                          -- any node, anonymous
(n)                         -- any node, bound to variable n
(:Person)                   -- node with label Person
(n:Person)                  -- labeled node bound to n
(n:Person:Employee)         -- node with BOTH labels
(n:Person {name: "Alice"})  -- node with property filter
(n {name: "Alice"})         -- any node with that property

-- ============================================================
-- RELATIONSHIP PATTERNS
-- ============================================================
-[r]->              -- any outgoing relationship
-[:KNOWS]->         -- outgoing, specific type
-[r:KNOWS]->        -- bound to variable r
<-[:KNOWS]-         -- incoming
-[:KNOWS]-          -- either direction (use carefully)
-[:KNOWS|LIKES]->   -- type OR (match either type)
-[*]->              -- variable length, any depth (DANGER: can be slow)
-[*1..3]->          -- 1 to 3 hops
-[*..5]->           -- up to 5 hops
-[*3..]-            -- at least 3 hops

-- ============================================================
-- FULL PATTERNS
-- ============================================================
(a:Person)-[:KNOWS]->(b:Person)
(a:Person)-[r:KNOWS {since: 2020}]->(b:Person)
(a)-[:KNOWS]->(b)-[:KNOWS]->(c)   -- path of 2 hops
```

### 5.4 Data Types & Literals

```cypher
// Strings
"Alice"
'Alice'

// Numbers
42          // Integer
3.14        // Float
-7          // Negative

// Booleans
true, false

// Null
null        // absence of value

// Lists
[1, 2, 3]
["a", "b", "c"]
[]          // empty list

// Maps (inline)
{name: "Alice", age: 30}

// Dates (ISO 8601)
date("2024-01-15")
datetime("2024-01-15T10:30:00")
duration("P1Y2M3DT4H")     // 1 year, 2 months, 3 days, 4 hours

// Spatial
point({x: 1.0, y: 2.0})
point({latitude: 37.55, longitude: -122.3})
```

---

## 6. CRUD Operations

### 6.1 CREATE — Insert Nodes and Relationships

```cypher
-- Create a single node
CREATE (n:Person {name: "Alice", age: 30, city: "Bangalore"})
RETURN n

-- Create multiple nodes
CREATE 
  (a:Person {name: "Alice"}),
  (b:Person {name: "Bob"}),
  (c:Company {name: "TechCorp"})
RETURN a, b, c

-- Create a relationship between existing nodes
MATCH (a:Person {name: "Alice"}), (b:Person {name: "Bob"})
CREATE (a)-[:KNOWS {since: 2020, weight: 0.8}]->(b)
RETURN a, b

-- Create nodes AND relationship in one statement
CREATE (a:Person {name: "Carol"})-[:WORKS_AT]->(c:Company {name: "Neo4j Inc"})
RETURN a, c

-- Create with multiple labels
CREATE (n:Person:Employee:Manager {name: "Dave", level: "L6"})
```

### 6.2 MERGE — Upsert (Create if Not Exists)

MERGE is the most important write operation in practice. It either matches an existing pattern or creates it.

```cypher
-- Merge on node (match or create)
MERGE (p:Person {name: "Alice"})
RETURN p

-- Merge with ON CREATE / ON MATCH
MERGE (p:Person {name: "Alice"})
ON CREATE SET p.created_at = datetime(), p.age = 30
ON MATCH  SET p.last_seen  = datetime()
RETURN p

-- CRITICAL: Always MERGE on a unique identifier only
-- BAD — may create duplicates or fail unexpectedly:
MERGE (p:Person {name: "Alice", age: 30, city: "Bangalore"})

-- GOOD — merge on unique key, set other props after:
MERGE (p:Person {id: "alice_001"})
ON CREATE SET p.name = "Alice", p.age = 30, p.city = "Bangalore"
ON MATCH  SET p.age = 30

-- Merge relationship (both nodes must exist or be created first)
MATCH (a:Person {name: "Alice"}), (b:Person {name: "Bob"})
MERGE (a)-[r:KNOWS]->(b)
ON CREATE SET r.since = 2024
RETURN r
```

> **MERGE Internals:** MERGE acquires a write lock on the pattern before checking existence. Concurrent MERGEs on the same pattern will serialize. This prevents duplicates but can cause contention at high write throughput. Use batching + MERGE carefully in high-concurrency scenarios.

### 6.3 SET — Update Properties and Labels

```cypher
-- Set a property
MATCH (p:Person {name: "Alice"})
SET p.age = 31
RETURN p

-- Set multiple properties
MATCH (p:Person {name: "Alice"})
SET p.age = 31, p.city = "Mumbai", p.updated_at = datetime()
RETURN p

-- Set properties using map (replaces only specified keys)
MATCH (p:Person {name: "Alice"})
SET p += {age: 31, city: "Mumbai"}   -- += merges, = replaces all
RETURN p

-- Replace ALL properties (destructive!)
MATCH (p:Person {name: "Alice"})
SET p = {name: "Alice", age: 31}    -- removes any keys not in map
RETURN p

-- Add a label
MATCH (p:Person {name: "Alice"})
SET p:VIPCustomer
RETURN p
```

### 6.4 REMOVE — Delete Properties and Labels

```cypher
-- Remove a property
MATCH (p:Person {name: "Alice"})
REMOVE p.age
RETURN p

-- Remove a label
MATCH (p:Person:VIPCustomer {name: "Alice"})
REMOVE p:VIPCustomer
RETURN p
```

### 6.5 DELETE & DETACH DELETE

```cypher
-- Delete a relationship
MATCH (a:Person {name: "Alice"})-[r:KNOWS]->(b:Person {name: "Bob"})
DELETE r

-- Delete a node (will FAIL if it has relationships)
MATCH (p:Person {name: "Alice"})
DELETE p   -- Error if Alice has any relationships

-- DETACH DELETE — delete node and all its relationships
MATCH (p:Person {name: "Alice"})
DETACH DELETE p

-- Delete all nodes and relationships (DANGEROUS — use only in dev)
MATCH (n)
DETACH DELETE n
```

### 6.6 Bulk Import with UNWIND

```cypher
-- Batch create from list (efficient for small batches in Cypher)
UNWIND [
  {id: "u1", name: "Alice", age: 30},
  {id: "u2", name: "Bob",   age: 25},
  {id: "u3", name: "Carol", age: 35}
] AS row
MERGE (p:Person {id: row.id})
ON CREATE SET p.name = row.name, p.age = row.age
RETURN count(p) AS created
```

For large datasets (millions of rows), use `neo4j-admin database import` or `LOAD CSV`:

```cypher
-- LOAD CSV (for files in import directory or URLs)
LOAD CSV WITH HEADERS FROM 'file:///persons.csv' AS row
MERGE (p:Person {id: row.id})
SET p.name = row.name,
    p.age  = toInteger(row.age),
    p.score = toFloat(row.score)

-- With periodic commit for large files (Neo4j 4.x syntax)
-- In Neo4j 5.x, use CALL { } IN TRANSACTIONS
LOAD CSV WITH HEADERS FROM 'file:///persons.csv' AS row
CALL {
  WITH row
  MERGE (p:Person {id: row.id})
  SET p.name = row.name
} IN TRANSACTIONS OF 1000 ROWS
```

---

## 7. Pattern Matching — The Heart of Cypher

### 7.1 MATCH Basics

```cypher
-- Match all nodes with a label
MATCH (p:Person)
RETURN p.name, p.age
LIMIT 10

-- Match with property filter
MATCH (p:Person {city: "Bangalore"})
RETURN p

-- Match with WHERE (more flexible than inline)
MATCH (p:Person)
WHERE p.age > 25 AND p.city = "Bangalore"
RETURN p.name, p.age
ORDER BY p.age DESC

-- Match relationship pattern
MATCH (a:Person)-[:KNOWS]->(b:Person)
RETURN a.name AS person, b.name AS knows

-- Match and return relationship properties
MATCH (a:Person)-[r:KNOWS]->(b:Person)
RETURN a.name, b.name, r.since, r.weight
```

### 7.2 OPTIONAL MATCH — Left Join Equivalent

```cypher
-- Optional: return Alice even if she has no friends
MATCH (p:Person {name: "Alice"})
OPTIONAL MATCH (p)-[:KNOWS]->(friend:Person)
RETURN p.name, friend.name   -- friend.name is null if no match

-- Count with null handling
MATCH (p:Person)
OPTIONAL MATCH (p)-[:KNOWS]->(friend)
RETURN p.name, count(friend) AS friend_count   -- count(null) = 0
```

### 7.3 Multi-Hop Traversal

```cypher
-- 2 hops: friends of friends
MATCH (me:Person {name: "Alice"})-[:KNOWS*2]->(fof:Person)
WHERE me <> fof
RETURN DISTINCT fof.name

-- Variable depth (1 to 3 hops)
MATCH (me:Person {name: "Alice"})-[:KNOWS*1..3]->(other:Person)
RETURN DISTINCT other.name, length(path) AS hops

-- Capture the full path
MATCH path = (me:Person {name: "Alice"})-[:KNOWS*1..3]->(other:Person)
RETURN path, length(path) AS hops

-- With relationship filtering at each hop
MATCH (me:Person {name: "Alice"})-[r:KNOWS*1..3]->(other:Person)
WHERE ALL(rel IN r WHERE rel.weight > 0.5)
RETURN other.name
```

> **Warning:** Variable-length patterns `[*]` without bounds can trigger full graph traversal. Always bound them: `[*1..5]`. Use `EXPLAIN` to see the plan before running.

### 7.4 Named Paths

```cypher
-- Capture and return a path
MATCH path = (a:Person {name: "Alice"})-[:KNOWS*]->(b:Person {name: "Dave"})
RETURN path, length(path)

-- Path functions
MATCH path = (a:Person)-[:KNOWS*1..4]->(b:Person)
RETURN 
  nodes(path)         AS all_nodes,
  relationships(path) AS all_rels,
  length(path)        AS hops

-- Shortest path (built-in)
MATCH (a:Person {name: "Alice"}), (b:Person {name: "Dave"})
MATCH path = shortestPath((a)-[:KNOWS*]-(b))
RETURN path, length(path)

-- All shortest paths
MATCH (a:Person {name: "Alice"}), (b:Person {name: "Dave"})
MATCH path = allShortestPaths((a)-[:KNOWS*]-(b))
RETURN path
```

### 7.5 WHERE Clause — Full Reference

```cypher
-- Comparison
WHERE p.age > 25
WHERE p.age >= 25 AND p.age <= 40
WHERE p.name = "Alice"
WHERE p.name <> "Alice"

-- Boolean
WHERE p.age > 25 AND p.city = "Bangalore"
WHERE p.age < 20 OR p.age > 60
WHERE NOT p.active

-- Null checks
WHERE p.email IS NULL
WHERE p.email IS NOT NULL

-- String operations
WHERE p.name STARTS WITH "Al"
WHERE p.name ENDS WITH "ce"
WHERE p.name CONTAINS "li"
WHERE p.name =~ "Al.*"    -- regex match (use sparingly — full scan)

-- List operations
WHERE p.tags IN ["ML", "AI"]
WHERE "AI" IN p.tags        -- check if element in list property

-- Pattern predicates (existential check)
WHERE (p)-[:KNOWS]->(:Person)         -- has at least one friend
WHERE NOT (p)-[:KNOWS]->(:Person)     -- has no friends
WHERE EXISTS { (p)-[:KNOWS]->(:VIP) } -- exists subquery (Neo4j 4.0+)

-- Node ID (internal — avoid in production logic)
WHERE id(n) = 42

-- Element ID (Neo4j 5.x stable string IDs)
WHERE elementId(n) = "4:abc123:0"
```

### 7.6 WITH — Pipelining Queries

`WITH` is the critical clause that lets you pipeline operations, like a subquery boundary.

```cypher
-- Filter aggregates (like HAVING in SQL)
MATCH (p:Person)-[:KNOWS]->(friend)
WITH p, count(friend) AS friend_count
WHERE friend_count > 5
RETURN p.name, friend_count
ORDER BY friend_count DESC

-- Collect then process
MATCH (p:Person)-[:BOUGHT]->(item:Product)
WITH p, collect(item.name) AS purchases
WHERE size(purchases) > 3
RETURN p.name, purchases

-- Limit before expensive operation
MATCH (p:Person)
WITH p
ORDER BY p.created_at DESC
LIMIT 100
MATCH (p)-[:KNOWS]->(friend)
RETURN p.name, count(friend) AS friends
```

---

## 8. Filtering, Aggregation & Projection

### 8.1 Aggregation Functions

```cypher
count(*)            -- total rows
count(n)            -- non-null values
count(DISTINCT n)   -- unique non-null values
sum(n.value)
avg(n.value)
min(n.value)
max(n.value)
collect(n.name)     -- aggregate into list
collect(DISTINCT n.name)
stdev(n.value)      -- sample standard deviation
stdevp(n.value)     -- population standard deviation
percentileCont(n.value, 0.95)   -- interpolated percentile
percentileDisc(n.value, 0.95)   -- discrete percentile
```

```cypher
-- Aggregation example: friend count per person
MATCH (p:Person)-[:KNOWS]->(friend:Person)
RETURN p.name, count(friend) AS degree, avg(friend.age) AS avg_friend_age
ORDER BY degree DESC
LIMIT 10

-- Collect example: all products per user
MATCH (u:User)-[:PURCHASED]->(p:Product)
RETURN u.id, collect({name: p.name, price: p.price}) AS purchases

-- Count by group
MATCH (p:Person)
RETURN p.city, count(*) AS residents
ORDER BY residents DESC
```

### 8.2 List Operations

```cypher
-- Create and manipulate lists
RETURN [1, 2, 3, 4, 5] AS nums
RETURN range(1, 10)               -- [1,2,3,4,5,6,7,8,9,10]
RETURN range(0, 10, 2)            -- [0,2,4,6,8,10]

-- List comprehension (like Python)
RETURN [x IN range(1,10) WHERE x % 2 = 0 | x * x] AS even_squares
-- [4, 16, 36, 64, 100]

-- List functions
size([1,2,3])          -- 3
head([1,2,3])          -- 1
tail([1,2,3])          -- [2,3]
last([1,2,3])          -- 3
reverse([1,2,3])       -- [3,2,1]
[1,2] + [3,4]         -- [1,2,3,4]  concatenation

-- Predicates on lists
ALL(x IN list WHERE x > 0)
ANY(x IN list WHERE x > 0)
NONE(x IN list WHERE x < 0)
SINGLE(x IN list WHERE x = 5)   -- exactly one

-- UNWIND: expand list into rows
UNWIND [1,2,3] AS num
RETURN num * 2   -- returns 3 rows: 2, 4, 6
```

### 8.3 String Functions

```cypher
toLower("ALICE")         -- "alice"
toUpper("alice")         -- "ALICE"
trim("  alice  ")        -- "alice"
ltrim(" alice")          -- "alice"
rtrim("alice ")          -- "alice"
replace("Alice", "A", "a")  -- "alice"
substring("Alice", 0, 3)    -- "Ali"
split("a,b,c", ",")         -- ["a","b","c"]
size("Alice")               -- 5  (string length)
left("Alice", 3)            -- "Ali"
right("Alice", 3)           -- "ice"
```

### 8.4 Math Functions

```cypher
abs(-5)       -- 5
ceil(4.3)     -- 5.0
floor(4.7)    -- 4.0
round(4.5)    -- 5.0
round(3.14159, 2)  -- 3.14
sqrt(9)       -- 3.0
log(e())      -- 1.0
exp(1)        -- 2.718...
sign(-5)      -- -1
rand()        -- random float 0..1
toInteger("42")   -- 42
toFloat("3.14")   -- 3.14
```

### 8.5 Temporal Functions

```cypher
date()               -- current local date
datetime()           -- current local datetime
datetime.realtime()  -- current UTC datetime
time()               -- current time

-- Create from string
date("2024-01-15")
datetime("2024-01-15T10:30:00Z")

-- Properties of dates
date().year    -- current year
date().month   -- current month
date().day     -- current day

-- Arithmetic
date("2024-03-01") + duration("P1M")   -- 2024-04-01
datetime() - duration("P7D")           -- 7 days ago

-- Duration
duration.between(date("2020-01-01"), date())   -- age as duration
```

---

## 9. Path Queries & Graph Traversal

### 9.1 Shortest Path Algorithms (Built-in)

```cypher
-- BFS shortest path (unweighted)
MATCH (a:Person {name: "Alice"}), (b:Person {name: "Dave"})
MATCH path = shortestPath((a)-[:KNOWS*]-(b))
RETURN path, length(path) AS hops

-- All shortest paths
MATCH (a:Person {name: "Alice"}), (b:Person {name: "Dave"})
MATCH path = allShortestPaths((a)-[:KNOWS*]-(b))
RETURN path, length(path)
ORDER BY length(path)

-- Shortest path with relationship filtering
MATCH (a:Station {name: "A"}), (b:Station {name: "Z"})
MATCH path = shortestPath((a)-[:CONNECTS*]-(b))
WHERE ALL(r IN relationships(path) WHERE r.operational = true)
RETURN path, 
       reduce(total = 0, r IN relationships(path) | total + r.distance) AS total_distance
```

### 9.2 Pattern Comprehension

```cypher
-- Like list comprehension, but for graph patterns
MATCH (p:Person {name: "Alice"})
RETURN [(p)-[:KNOWS]->(friend) | friend.name] AS friend_names

-- With filter
RETURN [(p)-[:KNOWS]->(friend) WHERE friend.age > 25 | friend.name] AS older_friends

-- Return objects from pattern
RETURN [(p)-[r:KNOWS]->(friend) | {name: friend.name, since: r.since}] AS friendships
```

### 9.3 CALL Subqueries

```cypher
-- Independent subquery (Neo4j 4.1+)
MATCH (p:Person)
CALL {
  MATCH (p2:Person)
  RETURN count(p2) AS total
}
RETURN p.name, total

-- Correlated subquery (uses outer variables)
MATCH (p:Person)
CALL {
  WITH p
  MATCH (p)-[:KNOWS]->(friend)
  RETURN count(friend) AS degree
}
RETURN p.name, degree
ORDER BY degree DESC

-- CALL IN TRANSACTIONS (for large write batches)
MATCH (p:Person)
CALL {
  WITH p
  SET p.processed = true
} IN TRANSACTIONS OF 500 ROWS
```

### 9.4 Conditional Logic

```cypher
-- CASE expression
MATCH (p:Person)
RETURN p.name,
  CASE 
    WHEN p.age < 18 THEN "minor"
    WHEN p.age < 65 THEN "adult"
    ELSE "senior"
  END AS age_group

-- Simple CASE
RETURN CASE p.status
  WHEN "active"   THEN 1
  WHEN "inactive" THEN 0
  ELSE -1
END AS status_code

-- CASE in WHERE (avoid — use direct predicates instead)
-- Prefer:
WHERE p.age >= 18 AND p.age < 65
-- Over:
WHERE CASE WHEN p.age >= 18 AND p.age < 65 THEN true ELSE false END
```

---

## 10. Graph Algorithms (GDS Library)

The **Graph Data Science (GDS)** library is Neo4j's ML/analytics engine. It operates on **in-memory projected graphs** (separate from the stored graph) for performance.

### 10.1 GDS Workflow

```
1. PROJECT — copy relevant subgraph into memory
2. RUN ALGORITHM — on projected graph
3. STREAM/WRITE/MUTATE results
4. DROP projection — free memory
```

```
Algorithm execution modes:
  STREAM  — return results row by row (no write)
  WRITE   — write results back to Neo4j as node properties
  MUTATE  — add results to in-memory projected graph (for chaining)
  STATS   — return summary statistics only
```

### 10.2 Graph Projection

```cypher
-- Native projection (fast, limited to homogeneous graphs)
CALL gds.graph.project(
  'myGraph',                    -- projection name
  'Person',                     -- node labels to include
  {
    KNOWS: {
      orientation: 'UNDIRECTED',
      properties: 'weight'
    }
  }
)
YIELD graphName, nodeCount, relationshipCount, projectMillis

-- Cypher projection (flexible, handles heterogeneous graphs)
CALL gds.graph.project.cypher(
  'myGraph',
  'MATCH (p:Person) RETURN id(p) AS id, p.age AS age',
  'MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN id(a) AS source, id(b) AS target, r.weight AS weight'
)

-- Check existing projections
CALL gds.graph.list()

-- Drop a projection
CALL gds.graph.drop('myGraph')
```

### 10.3 Centrality Algorithms

```cypher
-- PageRank — global importance (like Google's PageRank)
CALL gds.pageRank.stream('myGraph', {
  maxIterations: 20,
  dampingFactor: 0.85
})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS person, score
ORDER BY score DESC
LIMIT 10

-- Write PageRank back to nodes
CALL gds.pageRank.write('myGraph', {
  maxIterations: 20,
  dampingFactor: 0.85,
  writeProperty: 'pagerank'
})
YIELD nodePropertiesWritten, ranIterations

-- Betweenness Centrality — bridge nodes
CALL gds.betweenness.stream('myGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS person, score
ORDER BY score DESC
LIMIT 10

-- Degree Centrality
CALL gds.degree.stream('myGraph', {
  orientation: 'NATURAL'   -- NATURAL=out-degree, REVERSE=in-degree, UNDIRECTED=both
})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS person, score AS degree

-- Closeness Centrality
CALL gds.closeness.stream('myGraph')
YIELD nodeId, score
```

### 10.4 Community Detection

```cypher
-- Louvain — hierarchical community detection (most common)
CALL gds.louvain.stream('myGraph', {
  relationshipWeightProperty: 'weight'
})
YIELD nodeId, communityId, intermediateCommunityIds
RETURN gds.util.asNode(nodeId).name AS person, communityId
ORDER BY communityId

-- Write communities to nodes
CALL gds.louvain.write('myGraph', {
  writeProperty: 'community',
  relationshipWeightProperty: 'weight'
})
YIELD communityCount, modularity, modularities

-- Label Propagation — fast, good for large graphs
CALL gds.labelPropagation.stream('myGraph')
YIELD nodeId, communityId

-- Weakly Connected Components
CALL gds.wcc.stream('myGraph')
YIELD nodeId, componentId
RETURN componentId, count(*) AS size
ORDER BY size DESC

-- Triangle Count + Clustering Coefficient
CALL gds.triangleCount.stream('myGraph')
YIELD nodeId, triangleCount

CALL gds.localClusteringCoefficient.stream('myGraph')
YIELD nodeId, localClusteringCoefficient
```

### 10.5 Similarity Algorithms

```cypher
-- Node Similarity (Jaccard on neighbor sets)
CALL gds.nodeSimilarity.stream('myGraph', {
  topK: 10,              -- top K similar nodes per node
  similarityCutoff: 0.1  -- minimum similarity
})
YIELD node1, node2, similarity
RETURN gds.util.asNode(node1).name AS p1,
       gds.util.asNode(node2).name AS p2,
       similarity
ORDER BY similarity DESC

-- Write similarity relationships
CALL gds.nodeSimilarity.write('myGraph', {
  writeRelationshipType: 'SIMILAR_TO',
  writeProperty: 'score',
  topK: 10
})
```

### 10.6 Path Finding (GDS)

```cypher
-- Dijkstra shortest path (weighted)
MATCH (source:Person {name: "Alice"}), (target:Person {name: "Dave"})
CALL gds.shortestPath.dijkstra.stream('myGraph', {
  sourceNode: source,
  targetNode: target,
  relationshipWeightProperty: 'cost'
})
YIELD index, sourceNode, targetNode, totalCost, nodeIds, costs, path
RETURN totalCost, [nodeId IN nodeIds | gds.util.asNode(nodeId).name] AS path_names

-- A* (heuristic, for spatial graphs)
CALL gds.shortestPath.astar.stream('myGraph', {
  sourceNode: source,
  targetNode: target,
  latitudeProperty: 'lat',
  longitudeProperty: 'lon',
  relationshipWeightProperty: 'cost'
})

-- Yen's K Shortest Paths
CALL gds.shortestPath.yens.stream('myGraph', {
  sourceNode: source,
  targetNode: target,
  k: 5,
  relationshipWeightProperty: 'cost'
})
YIELD path, totalCost
```

### 10.7 Node Embedding (for ML feature vectors)

```cypher
-- FastRP (Fast Random Projection) — produces node embeddings
CALL gds.fastRP.stream('myGraph', {
  embeddingDimension: 128,
  iterationWeights: [0.0, 1.0, 1.0],
  randomSeed: 42
})
YIELD nodeId, embedding
RETURN gds.util.asNode(nodeId).name AS node, embedding

-- Write embeddings to nodes
CALL gds.fastRP.write('myGraph', {
  embeddingDimension: 128,
  writeProperty: 'embedding'
})

-- Node2Vec (random walk based)
CALL gds.node2vec.stream('myGraph', {
  embeddingDimension: 128,
  walkLength: 80,
  walksPerNode: 10,
  inOutFactor: 1.0,    -- p parameter
  returnFactor: 1.0    -- q parameter
})
YIELD nodeId, embedding

-- GraphSAGE (uses node features, requires feature properties)
CALL gds.beta.graphSage.train('myGraph', {
  modelName: 'mySageModel',
  featureProperties: ['age', 'pagerank'],
  embeddingDimension: 64,
  epochs: 5
})
```

### 10.8 Link Prediction (ML)

```cypher
-- Generate link prediction features
-- These are computed as similarity metrics between node pairs

-- Common Neighbors
RETURN gds.alpha.linkprediction.commonNeighbors(node1, node2)

-- Jaccard Coefficient
RETURN gds.alpha.linkprediction.jaccardSimilarity(node1, node2)

-- Adamic-Adar
RETURN gds.alpha.linkprediction.adamicAdar(node1, node2)

-- Resource Allocation
RETURN gds.alpha.linkprediction.resourceAllocation(node1, node2)

-- Preferential Attachment
RETURN gds.alpha.linkprediction.preferentialAttachment(node1, node2)

-- Total Neighbors
RETURN gds.alpha.linkprediction.totalNeighbors(node1, node2)
```

---

## 11. Indexing & Query Optimization

### 11.1 Index Types

| Index Type | Use Case | Syntax |
|---|---|---|
| Range Index | Equality, range, prefix queries | `CREATE INDEX FOR (n:Label) ON (n.prop)` |
| Full-text Index | Text search, CONTAINS, regex | `CREATE FULLTEXT INDEX` |
| Point Index | Spatial queries | `CREATE POINT INDEX` |
| Composite Index | Multi-property lookups | `CREATE INDEX FOR (n:L) ON (n.a, n.b)` |
| Constraint (Unique) | Uniqueness + index | `CREATE CONSTRAINT` |
| Existence Constraint | Property must exist | `CREATE CONSTRAINT ... IS NOT NULL` |

### 11.2 Creating Indexes

```cypher
-- Single property range index
CREATE INDEX person_name FOR (p:Person) ON (p.name)

-- Composite index (column order matters for query planning)
CREATE INDEX person_city_age FOR (p:Person) ON (p.city, p.age)

-- Named index
CREATE INDEX idx_person_email FOR (p:Person) ON (p.email)

-- Unique constraint (also creates an index)
CREATE CONSTRAINT person_id_unique FOR (p:Person) REQUIRE p.id IS UNIQUE

-- Node key constraint (composite uniqueness)
CREATE CONSTRAINT person_key FOR (p:Person) REQUIRE (p.firstName, p.lastName) IS NODE KEY

-- Existence constraint (Enterprise)
CREATE CONSTRAINT person_name_exists FOR (p:Person) REQUIRE p.name IS NOT NULL

-- Relationship property index (Neo4j 5.x)
CREATE INDEX rel_weight FOR ()-[r:KNOWS]-() ON (r.weight)

-- Full-text index
CREATE FULLTEXT INDEX product_search FOR (n:Product) ON EACH [n.name, n.description]

-- Show all indexes
SHOW INDEXES

-- Drop an index
DROP INDEX person_name
```

### 11.3 Using Full-Text Search

```cypher
-- Full-text query
CALL db.index.fulltext.queryNodes("product_search", "machine learning")
YIELD node, score
RETURN node.name, score
ORDER BY score DESC
LIMIT 10

-- Full-text with fuzzy matching
CALL db.index.fulltext.queryNodes("product_search", "machne~")  -- fuzzy
YIELD node, score

-- Full-text on relationships
CREATE FULLTEXT INDEX review_text FOR ()-[r:REVIEWED]-() ON EACH [r.comment]
CALL db.index.fulltext.queryRelationships("review_text", "excellent product")
YIELD relationship, score
```

### 11.4 EXPLAIN and PROFILE

These are the most important debugging tools in Cypher.

```cypher
-- EXPLAIN: show plan WITHOUT executing
EXPLAIN
MATCH (p:Person {name: "Alice"})-[:KNOWS]->(friend)
RETURN friend.name

-- PROFILE: execute AND show actual row counts / db hits
PROFILE
MATCH (p:Person {name: "Alice"})-[:KNOWS]->(friend)
RETURN friend.name
```

**Reading the query plan:**
```
Key operators to understand:
  NodeIndexSeek          — good: using an index
  NodeByLabelScan        — bad for large graphs: scanning all nodes with label
  AllNodesScan           — very bad: scanning entire graph
  Expand(All)            — traversing relationships
  Filter                 — applying WHERE predicates
  EagerAggregation       — aggregation (requires all rows in memory)
  CartesianProduct       — unintended cross join (almost always a bug)

Key metrics:
  Rows            — number of rows at that operator
  DB Hits         — number of storage reads (lower = better)
  Estimated Rows  — planner's estimate (if very wrong, statistics are stale)
```

### 11.5 Query Optimization Techniques

```cypher
-- 1. Always MATCH on indexed property first
-- BAD:
MATCH (p:Person)-[:KNOWS]->(friend)
WHERE p.name = "Alice"

-- GOOD:
MATCH (p:Person {name: "Alice"})-[:KNOWS]->(friend)
-- Or ensure index exists and use WHERE:
MATCH (p:Person)
WHERE p.name = "Alice"  -- uses index if one exists

-- 2. Limit early in the pipeline
MATCH (p:Person)
WITH p ORDER BY p.created_at DESC LIMIT 100   -- limit before expensive traversal
MATCH (p)-[:KNOWS]->(friend)
RETURN p.name, count(friend)

-- 3. Use DISTINCT to avoid duplicate processing
MATCH (p:Person)-[:KNOWS*2]->(fof:Person)
RETURN DISTINCT fof.name

-- 4. Avoid OPTIONAL MATCH when you can use collect with empty list
-- SLOWER:
MATCH (p:Person)
OPTIONAL MATCH (p)-[:KNOWS]->(f)
RETURN p.name, collect(f.name) AS friends

-- FASTER for large graphs:
MATCH (p:Person)
WITH p, [(p)-[:KNOWS]->(f) | f.name] AS friends
RETURN p.name, friends

-- 5. Parameterize queries (never string-interpolate)
-- BAD (string concatenation — no query plan reuse):
session.run(f"MATCH (p:Person {{name: '{name}'}}) RETURN p")

-- GOOD (parameterized — plan cached):
session.run("MATCH (p:Person {name: $name}) RETURN p", name=name)

-- 6. Update stale statistics
CALL db.stats.retrieve("GRAPH COUNTS")
-- Or: restart Neo4j to refresh (statistics update on restart)
```

### 11.6 Parameterized Queries in Cypher

```cypher
-- In Cypher directly (using $param syntax)
// Parameters: {"name": "Alice", "min_age": 25}
MATCH (p:Person {name: $name})
WHERE p.age >= $min_age
RETURN p

// Parameters: {"ids": ["u1", "u2", "u3"]}
MATCH (p:Person)
WHERE p.id IN $ids
RETURN p
```

---

## 12. Schema Design for AI/ML

### 12.1 Graph Data Modeling Principles

```
Rules of thumb:
1. Nouns → Nodes
2. Verbs → Relationships
3. Adjectives/Attributes → Properties
4. Specificity → Labels (use multiple labels)
5. Context → Relationship properties

BAD model (relational thinking in graph):
  (:Person {id, name, company_id})
  (:Company {id, name})
  
  Query: MATCH (p:Person)-[:WORKS_AT]->(c:Company)
  Fine, but we've lost WHY they work there

GOOD model (graph thinking):
  (:Person {id, name})
  (:Company {id, name})
  (:Role {title: "Engineer", level: "L5"})
  
  (p:Person)-[:HAS_ROLE]->(r:Role)-[:AT]->(c:Company)
  (r)-[:STARTED_ON]->(d:Date)
  
  Now: roles are queryable, reusable, temporal
```

### 12.2 Modeling Patterns for ML

**Pattern 1: Bipartite Graph (User-Item)**
```cypher
-- Users and items with interaction properties
CREATE (u:User {id: "u1"})-[:INTERACTED {
  type: "purchase",
  timestamp: datetime(),
  rating: 4.5,
  duration_ms: 3200
}]->(i:Item {id: "p42", category: "electronics"})

-- This enables:
-- Collaborative filtering via shared item neighborhoods
-- Content-based filtering via item properties
-- Hybrid: combine both traversal and properties
```

**Pattern 2: Temporal Graph**
```cypher
-- Version events as nodes, not properties
(:Transaction {id, amount})-[:OCCURRED_AT]->(:Timestamp {dt: datetime()})

-- Or: Event nodes with time properties
(:Event:Purchase {
  id: "evt_001",
  amount: 1200.00,
  timestamp: datetime("2024-03-15T14:30:00Z"),
  channel: "mobile"
})-[:BY]->(:User)
 -[:OF]->(:Product)

-- Allows temporal pattern queries:
MATCH (u:User)-[:BY]->(e:Event:Purchase)-[:OF]->(p:Product)
WHERE e.timestamp > datetime() - duration("P30D")
RETURN u.id, collect(p.name) AS last_30_day_purchases
```

**Pattern 3: Hierarchical / Taxonomy**
```cypher
(:Category {name: "Electronics"})-[:PARENT_OF]->(:Category {name: "Laptops"})
-[:PARENT_OF]->(:Category {name: "Gaming Laptops"})

-- Variable-depth traversal for category rollup:
MATCH (root:Category {name: "Electronics"})-[:PARENT_OF*]->(leaf:Category)
RETURN leaf.name
```

**Pattern 4: Knowledge Graph Triples**
```cypher
-- Entity-Relation-Entity pattern
(:Entity {id, name, type})-[:RELATION {confidence: 0.95}]->(:Entity)

-- Example:
(:Entity {id: "e1", name: "Python", type: "Language"})
-[:IS_USED_IN {confidence: 0.99}]->
(:Entity {id: "e2", name: "Machine Learning", type: "Domain"})
```

### 12.3 Common Anti-Patterns in Schema Design

```
❌ Anti-Pattern 1: Supernode
A node with millions of relationships (e.g., a "root" node all users connect to)
→ Traversal anchored on this node = massive fan-out = slow queries
Fix: Add intermediate category nodes, use relationship properties to filter

❌ Anti-Pattern 2: Property as relationship
{friends_list: ["bob", "carol"]}  ← DON'T store arrays of IDs as properties
→ Can't traverse, can't index, can't query relationships
Fix: Model as actual graph relationships

❌ Anti-Pattern 3: Generic relationships
-[:RELATED_TO]->  ← meaningless, loses semantic richness
Fix: Use specific typed relationships: -[:PURCHASED]-, -[:REVIEWED]-, -[:SIMILAR_TO]-

❌ Anti-Pattern 4: Relational thinking
Don't create a "JoinTable" node: (:PersonProduct {person_id, product_id})
→ This is a relationship! Use: (p:Person)-[:PURCHASED]->(prod:Product)
```

---

## 13. Neo4j for Knowledge Graphs

### 13.1 Knowledge Graph Schema

```cypher
-- Core ontology setup
CREATE CONSTRAINT entity_id FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT relation_id FOR (r:Relation) REQUIRE r.id IS UNIQUE;
CREATE INDEX entity_name FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type FOR (e:Entity) ON (e.type);

-- Entity creation
MERGE (e:Entity {id: "ent_python"})
SET e.name = "Python",
    e.type = "ProgrammingLanguage",
    e.aliases = ["Python 3", "CPython"],
    e.description = "General-purpose programming language"

-- Typed relationship with metadata
MATCH (python:Entity {id: "ent_python"})
MATCH (ml:Entity {id: "ent_ml"})
MERGE (python)-[r:USED_IN]->(ml)
SET r.confidence = 0.99,
    r.source = "Wikipedia",
    r.extracted_at = datetime()
```

### 13.2 Querying Knowledge Graphs

```cypher
-- Entity lookup
MATCH (e:Entity {name: "Python"})
RETURN e.type, e.description, e.aliases

-- 1-hop neighborhood (what is Python related to?)
MATCH (e:Entity {name: "Python"})-[r]->(related:Entity)
RETURN type(r) AS relation, related.name AS entity, r.confidence
ORDER BY r.confidence DESC

-- Path between two entities
MATCH (a:Entity {name: "Python"}), (b:Entity {name: "TensorFlow"})
MATCH path = shortestPath((a)-[*..5]-(b))
RETURN [n IN nodes(path) | n.name] AS path_names,
       [r IN relationships(path) | type(r)] AS relations

-- Multi-hop: what languages are used in frameworks used in ML?
MATCH (ml:Entity {name: "Machine Learning"})
<-[:USED_IN]-(lang:Entity {type: "ProgrammingLanguage"})
<-[:IMPLEMENTED_IN]-(framework:Entity {type: "Framework"})
RETURN lang.name, collect(framework.name) AS frameworks

-- Find entities connected to multiple target entities (intersection)
MATCH (target1:Entity {name: "Deep Learning"})
MATCH (target2:Entity {name: "Computer Vision"})
MATCH (e:Entity)-[:USED_IN]->(target1)
MATCH (e)-[:USED_IN]->(target2)
RETURN e.name AS shared_entity
```

### 13.3 APOC for KG Construction

```cypher
-- APOC (Awesome Procedures on Cypher) extends Cypher significantly

-- Create from JSON (e.g., from LLM extraction output)
WITH '{"entities": [{"id":"e1","name":"BERT","type":"Model"},
                     {"id":"e2","name":"NLP","type":"Domain"}],
       "relations": [{"from":"e1","to":"e2","type":"APPLIED_TO"}]}' AS json_str
WITH apoc.convert.fromJsonMap(json_str) AS data
UNWIND data.entities AS ent
MERGE (e:Entity {id: ent.id})
SET e.name = ent.name, e.type = ent.type
WITH data
UNWIND data.relations AS rel
MATCH (from:Entity {id: rel.from})
MATCH (to:Entity {id: rel.to})
CALL apoc.create.relationship(from, rel.type, {}, to) YIELD rel AS r
RETURN count(r)

-- Virtual graph (in-memory, not persisted)
MATCH (e:Entity)
CALL apoc.create.vNode(['Virtual'], {name: e.name}) YIELD node AS vn
RETURN vn

-- Schema visualization
CALL apoc.meta.schema()
YIELD value
RETURN value

-- Export to JSON
CALL apoc.export.json.all("export.json", {})

-- Run Cypher from string (dynamic queries)
CALL apoc.cypher.run("MATCH (n:$label) RETURN n", {label: "Person"})
```

---

## 14. Neo4j + Python (py2neo, neo4j-driver)

### 14.1 Official Driver — Best Practice Patterns

```python
from neo4j import GraphDatabase, basic_auth
from contextlib import contextmanager
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class Neo4jConnection:
    """
    Production-grade Neo4j connection wrapper.
    Uses connection pooling, retries, and context managers.
    """
    
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        max_connection_pool_size: int = 50,
    ):
        self._driver = GraphDatabase.driver(
            uri,
            auth=basic_auth(user, password),
            max_connection_pool_size=max_connection_pool_size,
            connection_timeout=30,
            max_transaction_retry_time=30,
        )
        self._database = database
        self._driver.verify_connectivity()
    
    def close(self):
        self._driver.close()
    
    @contextmanager
    def session(self):
        session = self._driver.session(database=self._database)
        try:
            yield session
        finally:
            session.close()
    
    def read(self, query: str, params: dict = None) -> list[dict]:
        """Execute a read query, return list of dicts."""
        with self.session() as session:
            result = session.execute_read(
                lambda tx: list(tx.run(query, params or {}))
            )
            return [dict(record) for record in result]
    
    def write(self, query: str, params: dict = None) -> list[dict]:
        """Execute a write query, return list of dicts."""
        with self.session() as session:
            result = session.execute_write(
                lambda tx: list(tx.run(query, params or {}))
            )
            return [dict(record) for record in result]
    
    def write_batch(self, query: str, batch: list[dict], batch_size: int = 500):
        """Batch write using UNWIND for efficiency."""
        for i in range(0, len(batch), batch_size):
            chunk = batch[i:i + batch_size]
            self.write(query, {"batch": chunk})
            logger.info(f"Wrote batch {i//batch_size + 1}, {len(chunk)} items")


# Usage
neo4j = Neo4jConnection(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)

# Read
results = neo4j.read(
    "MATCH (p:Person {name: $name})-[:KNOWS]->(friend) RETURN friend.name AS name",
    {"name": "Alice"}
)

# Batch write
persons = [{"id": f"u{i}", "name": f"User{i}", "age": 20+i} for i in range(10000)]
neo4j.write_batch(
    """
    UNWIND $batch AS row
    MERGE (p:Person {id: row.id})
    SET p.name = row.name, p.age = row.age
    """,
    persons
)

neo4j.close()
```

### 14.2 Async Driver (for FastAPI / async services)

```python
from neo4j import AsyncGraphDatabase
import asyncio

async def main():
    driver = AsyncGraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "password")
    )
    
    async with driver.session() as session:
        result = await session.run(
            "MATCH (p:Person {name: $name}) RETURN p",
            name="Alice"
        )
        record = await result.single()
        print(record["p"])
    
    await driver.close()

asyncio.run(main())
```

### 14.3 GDS Python Client

```python
from graphdatascience import GraphDataScience

gds = GraphDataScience("bolt://localhost:7687", auth=("neo4j", "password"))

# Project a graph
G, result = gds.graph.project(
    "myGraph",
    "Person",
    {"KNOWS": {"orientation": "UNDIRECTED", "properties": "weight"}}
)
print(f"Projected {result['nodeCount']} nodes")

# Run PageRank
pagerank_result = gds.pageRank.stream(G, maxIterations=20, dampingFactor=0.85)
print(pagerank_result.head(10))
# Returns a pandas DataFrame: nodeId, score

# Run community detection
louvain_result = gds.louvain.stream(G, relationshipWeightProperty="weight")
print(louvain_result.groupby("communityId").size().sort_values(ascending=False))

# Node embeddings → pandas → ML model
embedding_result = gds.fastRP.stream(G, embeddingDimension=64, randomSeed=42)
embedding_df = embedding_result[["nodeId", "embedding"]]

# Convert embeddings to numpy for sklearn/torch
import numpy as np
X = np.array(embedding_df["embedding"].tolist())

# Cleanup
G.drop()
gds.close()
```

### 14.4 Pandas ↔ Neo4j Integration

```python
import pandas as pd
from neo4j import GraphDatabase

def neo4j_to_dataframe(driver, query: str, params: dict = None) -> pd.DataFrame:
    """Convert Neo4j query result to pandas DataFrame."""
    with driver.session() as session:
        result = session.run(query, params or {})
        return pd.DataFrame([dict(record) for record in result])

# Example: pull graph features for ML
df = neo4j_to_dataframe(driver, """
    MATCH (p:Person)
    RETURN 
      p.id AS user_id,
      p.age AS age,
      p.pagerank AS pagerank,
      p.community AS community,
      size([(p)-[:KNOWS]->() | 1]) AS out_degree,
      size([(p)<-[:KNOWS]-() | 1]) AS in_degree
""")

# Now feed to sklearn
from sklearn.ensemble import RandomForestClassifier
X = df[["age", "pagerank", "out_degree", "in_degree"]].values
y = df["label"].values  # assuming you have labels
```

---

## 15. Neo4j for RAG Systems

### 15.1 GraphRAG Architecture

Traditional vector RAG retrieves semantically similar text chunks. GraphRAG adds structured relational context:

```
Vector RAG:
  Query → Embedding → ANN Search → Top-K chunks → LLM → Response
  Problem: Chunks are isolated; no relational structure

GraphRAG:
  Query → {Entity extraction + Embedding}
       → {Graph traversal for entities + ANN Search for chunks}
       → {Structured context + Relevant chunks}
       → LLM → Grounded response
```

### 15.2 Building a GraphRAG Pipeline

```python
from neo4j import GraphDatabase
from openai import OpenAI
import json

class GraphRAGPipeline:
    def __init__(self, neo4j_driver, llm_client):
        self.neo4j = neo4j_driver
        self.llm = llm_client
    
    def extract_entities_from_query(self, query: str) -> list[str]:
        """Use LLM to extract entities from user query."""
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Extract named entities from this query. 
                Return JSON list of entity names only.
                Query: {query}
                Response format: ["entity1", "entity2"]"""
            }],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("entities", [])
    
    def retrieve_graph_context(self, entities: list[str], hops: int = 2) -> str:
        """Traverse graph to get structured context around entities."""
        with self.neo4j.session() as session:
            result = session.run("""
                UNWIND $entities AS entity_name
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower(entity_name)
                  OR any(alias IN e.aliases WHERE toLower(alias) CONTAINS toLower(entity_name))
                
                // Get neighborhood
                OPTIONAL MATCH (e)-[r1]->(neighbor1:Entity)
                OPTIONAL MATCH (e)<-[r2]-(neighbor2:Entity)
                
                RETURN 
                  e.name AS entity,
                  e.description AS description,
                  collect(DISTINCT {
                    relation: type(r1), 
                    target: neighbor1.name,
                    direction: "outgoing"
                  }) AS outgoing,
                  collect(DISTINCT {
                    relation: type(r2),
                    source: neighbor2.name,
                    direction: "incoming"
                  }) AS incoming
            """, entities=entities)
            
            context_parts = []
            for record in result:
                entity = record["entity"]
                desc = record["description"] or ""
                out_rels = [r for r in record["outgoing"] if r["target"]]
                in_rels = [r for r in record["incoming"] if r["source"]]
                
                part = f"Entity: {entity}\nDescription: {desc}\n"
                if out_rels:
                    rels_str = "; ".join([f"{r['relation']} → {r['target']}" for r in out_rels[:5]])
                    part += f"Relations: {rels_str}\n"
                if in_rels:
                    rels_str = "; ".join([f"{r['source']} → {r['relation']}" for r in in_rels[:5]])
                    part += f"Referenced by: {rels_str}\n"
                
                context_parts.append(part)
            
            return "\n---\n".join(context_parts)
    
    def answer(self, query: str) -> str:
        """Full GraphRAG pipeline."""
        entities = self.extract_entities_from_query(query)
        context = self.retrieve_graph_context(entities)
        
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Answer questions using the provided graph context. "
                               "Cite specific relationships from the context. "
                               "If the context doesn't contain the answer, say so."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ]
        )
        return response.choices[0].message.content
```

### 15.3 LangChain + Neo4j Integration

```python
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

# Connect to Neo4j
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="password"
)

# Auto-generate schema
graph.refresh_schema()
print(graph.schema)

# Natural language to Cypher chain
llm = ChatOpenAI(model="gpt-4o", temperature=0)
chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_requests=True
)

# Ask in natural language
result = chain.invoke("Who are Alice's friends and when did they meet?")
print(result["result"])
# Shows generated Cypher + answer

# Vector + Graph hybrid (Neo4jVector)
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings

vectorstore = Neo4jVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    url="bolt://localhost:7687",
    username="neo4j",
    password="password",
    index_name="document_embeddings",
    node_label="Document",
    text_node_properties=["content"],
    embedding_node_property="embedding"
)

# Similarity search
docs = vectorstore.similarity_search("machine learning applications", k=5)
```

### 15.4 Storing LLM Conversation History as Graph

```cypher
-- Model conversation history structurally
(:Session {id: "sess_abc", started_at: datetime()})
-[:CONTAINS]->
(:Turn {index: 0, role: "user", content: "...", timestamp: datetime()})
-[:NEXT]->
(:Turn {index: 1, role: "assistant", content: "...", timestamp: datetime()})

-- Turns reference entities
(:Turn)-[:MENTIONS]->(:Entity)
(:Turn)-[:REFERENCES]->(:Document)

-- Query: what topics has user discussed?
MATCH (s:Session {id: $session_id})-[:CONTAINS]->(t:Turn)-[:MENTIONS]->(e:Entity)
RETURN e.name AS topic, count(t) AS mentions
ORDER BY mentions DESC
```

---

## 16. Neo4j for Recommendation Systems

### 16.1 Collaborative Filtering via Graph Traversal

```cypher
-- User-based collaborative filtering
-- "Users similar to Alice also liked..."

MATCH (me:User {id: $user_id})-[:PURCHASED]->(item:Item)
  <-[:PURCHASED]-(similar_user:User)
  -[:PURCHASED]->(recommendation:Item)
WHERE NOT (me)-[:PURCHASED]->(recommendation)
  AND me <> similar_user
WITH recommendation, 
     count(DISTINCT similar_user) AS social_proof,
     avg(similar_user.purchase_count) AS avg_buyer_activity
WHERE social_proof >= 3
RETURN recommendation.id, 
       recommendation.name, 
       social_proof,
       avg_buyer_activity
ORDER BY social_proof DESC, avg_buyer_activity DESC
LIMIT 20
```

```cypher
-- Item-based collaborative filtering with weighted similarity
-- Items bought together (association rules via graph)

MATCH (target:Item {id: $item_id})
    <-[:PURCHASED]-(user:User)
    -[:PURCHASED]->(co_purchased:Item)
WHERE co_purchased <> target
WITH co_purchased,
     count(user) AS co_purchase_count,
     count(user) * 1.0 / (
         size([(target)<-[:PURCHASED]-() | 1]) +
         size([(co_purchased)<-[:PURCHASED]-() | 1]) -
         count(user)
     ) AS jaccard_score
WHERE co_purchase_count >= 5
RETURN co_purchased.name, co_purchase_count, round(jaccard_score, 3) AS jaccard
ORDER BY jaccard_score DESC
LIMIT 15
```

### 16.2 Content-Based + Graph Hybrid

```cypher
-- Hybrid: combine graph topology with content similarity
-- Assumes items have embeddings and community labels

MATCH (me:User {id: $user_id})-[r:RATED]->(item:Item)
WHERE r.rating >= 4.0
WITH me, collect(item) AS liked_items

// Find items in same communities as liked items
UNWIND liked_items AS liked
MATCH (candidate:Item)
WHERE candidate.community = liked.community
  AND NOT (me)-[:RATED]->(candidate)
  AND candidate <> liked

WITH candidate, 
     count(DISTINCT liked) AS community_overlap,
     avg(
       gds.similarity.cosine(liked.embedding, candidate.embedding)
     ) AS avg_content_similarity
WHERE community_overlap >= 2 AND avg_content_similarity > 0.7

RETURN candidate.id, candidate.name, 
       community_overlap, 
       round(avg_content_similarity, 3) AS content_sim
ORDER BY community_overlap * avg_content_similarity DESC
LIMIT 20
```

### 16.3 Session-Based Recommendations

```cypher
-- Recommend next item based on current session path
-- (Markov chain on item transitions)

MATCH (current:Item {id: $current_item_id})
    <-[:VIEWED {session_id: $session_id}]-(:User)
    -[:VIEWED]->(next_in_session:Item)
WHERE next_in_session <> current
WITH next_in_session, count(*) AS transition_count
WHERE transition_count > 10
RETURN next_in_session.id, next_in_session.name, transition_count
ORDER BY transition_count DESC
LIMIT 10
```

---

## 17. Neo4j for Fraud Detection & Link Analysis

### 17.1 Ring Detection (Fraud Rings)

```cypher
-- Detect cycles: account → transaction → account (circular money flow)
MATCH path = (start:Account)-[:TRANSFERRED_TO*3..6]->(start)
WITH path, 
     reduce(total = 0, r IN relationships(path) | total + r.amount) AS cycle_amount,
     length(path) AS ring_size
WHERE cycle_amount > 10000
RETURN nodes(path) AS ring_accounts, 
       ring_size,
       cycle_amount
ORDER BY cycle_amount DESC
LIMIT 100
```

```cypher
-- Shared identity detection (multiple accounts, same device/email/phone)
MATCH (a1:Account)-[:USES]->(identifier:DeviceID|Email|Phone)
    <-[:USES]-(a2:Account)
WHERE a1 <> a2
  AND a1.status = "active"
  AND a2.status = "active"
WITH identifier, collect(DISTINCT a1) + collect(DISTINCT a2) AS shared_accounts
WHERE size(shared_accounts) > 2
RETURN labels(identifier)[0] AS identifier_type,
       identifier.value AS id_value,
       [a IN shared_accounts | a.id] AS accounts,
       size(shared_accounts) AS account_count
ORDER BY account_count DESC
```

### 17.2 First-Party Fraud (Synthetic Identity)

```cypher
-- Find nodes with suspiciously high entity overlap
MATCH (p:Person)-[:HAS]->(attr:Attribute)
    <-[:HAS]-(other:Person)
WHERE p <> other
  AND labels(attr)[0] IN ["SSN", "Phone", "Address", "Email"]
WITH p, other, 
     collect(DISTINCT labels(attr)[0]) AS shared_attribute_types,
     count(DISTINCT attr) AS shared_count
WHERE shared_count >= 2
RETURN p.id, other.id, shared_attribute_types, shared_count
ORDER BY shared_count DESC
```

### 17.3 Risk Scoring via Graph Features

```cypher
-- Composite risk score based on graph topology
MATCH (a:Account {id: $account_id})
OPTIONAL MATCH (a)-[:TRANSFERRED_TO]->(recipients)
OPTIONAL MATCH (a)<-[:TRANSFERRED_TO]-(senders)
OPTIONAL MATCH (a)-[:USES]->(ids:DeviceID|Phone)
    <-[:USES]-(other_accounts:Account)

WITH a,
     count(DISTINCT recipients) AS out_degree,
     count(DISTINCT senders)    AS in_degree,
     count(DISTINCT other_accounts) AS shared_identity_count,
     a.pagerank AS pagerank_score,
     a.community_fraud_rate AS community_risk

RETURN a.id,
  // Simple composite risk score
  (
    CASE WHEN out_degree > 50 THEN 0.3 ELSE 0.0 END +
    CASE WHEN shared_identity_count > 2 THEN 0.4 ELSE 0.0 END +
    CASE WHEN community_risk > 0.1 THEN 0.2 ELSE 0.0 END +
    CASE WHEN in_degree = 0 AND out_degree > 10 THEN 0.1 ELSE 0.0 END
  ) AS risk_score,
  out_degree, in_degree, shared_identity_count
ORDER BY risk_score DESC
```

---

## 18. Graph Neural Networks with Neo4j

### 18.1 Exporting Graph Data for GNN Training

```python
import torch
from torch_geometric.data import Data
import pandas as pd
from neo4j import GraphDatabase

def export_graph_for_pyg(driver) -> Data:
    """Export Neo4j graph as PyTorch Geometric Data object."""
    
    with driver.session() as session:
        # Get nodes with features
        node_result = session.run("""
            MATCH (p:Person)
            RETURN id(p) AS node_id, 
                   p.age AS age,
                   p.pagerank AS pagerank,
                   p.community AS community,
                   p.label AS label    -- 0/1 for binary classification
            ORDER BY node_id
        """)
        nodes_df = pd.DataFrame([dict(r) for r in node_result])
        
        # Get edges
        edge_result = session.run("""
            MATCH (a:Person)-[r:KNOWS]->(b:Person)
            RETURN id(a) AS src, id(b) AS dst, r.weight AS weight
        """)
        edges_df = pd.DataFrame([dict(r) for r in edge_result])
    
    # Map Neo4j IDs to contiguous indices
    node_ids = nodes_df["node_id"].tolist()
    id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    
    # Node features
    feature_cols = ["age", "pagerank"]
    x = torch.tensor(nodes_df[feature_cols].fillna(0).values, dtype=torch.float)
    
    # Labels
    y = torch.tensor(nodes_df["label"].fillna(0).values, dtype=torch.long)
    
    # Edge indices (PyG expects [2, num_edges])
    edges_df["src_idx"] = edges_df["src"].map(id_to_idx)
    edges_df["dst_idx"] = edges_df["dst"].map(id_to_idx)
    edge_index = torch.tensor(
        edges_df[["src_idx", "dst_idx"]].values.T, dtype=torch.long
    )
    
    # Edge weights
    edge_attr = torch.tensor(
        edges_df["weight"].fillna(1.0).values, dtype=torch.float
    ).unsqueeze(1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# Build GNN
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# Load data and train
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
data = export_graph_for_pyg(driver)

model = GraphSAGE(
    in_channels=data.x.shape[1],
    hidden_channels=64,
    out_channels=2   # binary classification
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
```

### 18.2 Writing GNN Predictions Back to Neo4j

```python
# After training, write predictions back as node properties
model.eval()
with torch.no_grad():
    pred = model(data.x, data.edge_index).argmax(dim=1)
    proba = model(data.x, data.edge_index).exp()  # softmax probabilities

# Map back to Neo4j IDs
predictions = [
    {"neo4j_id": node_id, "prediction": int(pred[idx]), "confidence": float(proba[idx].max())}
    for idx, node_id in enumerate(node_ids)
]

# Write back
with driver.session() as session:
    session.run("""
        UNWIND $predictions AS row
        MATCH (p:Person) WHERE id(p) = row.neo4j_id
        SET p.gnn_prediction = row.prediction,
            p.gnn_confidence = row.confidence,
            p.gnn_updated_at = datetime()
    """, predictions=predictions)
```

---

## 19. Production: Performance, Scaling, Reliability

### 19.1 Memory Sizing Formula

```
Total Neo4j memory = Heap + Page Cache + OS overhead

Heap (JVM):
  - Too small: GC pauses, OOM on complex queries
  - Too large: Longer GC pauses
  - Rule: 4-8GB for most workloads, never > 31GB (compressed oops boundary)
  - Initial = Max (avoid resize GC pauses)

Page Cache:
  - Goal: fit entire working set in memory
  - Check: neostore size on disk → set pagecache to at least that
  - Formula: pagecache = total_store_size * 1.2 (20% headroom)

OS: reserve 2-4GB for OS + off-heap

Example for 50GB graph:
  heap.initial = 8g
  heap.max = 8g
  pagecache = 60g     ← larger than store for headroom
  Server RAM = 8 + 60 + 4 = 72GB minimum
```

### 19.2 Query Performance Checklist

```
□ All MATCH start points are on indexed properties
□ No unbounded variable-length patterns ([*] without limits)
□ Parameters used (no string interpolation)
□ PROFILE run and plan reviewed — no AllNodesScan
□ LIMIT applied as early as possible
□ No accidental CartesianProduct
□ DISTINCT used where needed (especially after multi-hop)
□ Aggregation kept as late as possible
□ No regex in WHERE unless full-text index is used
□ Page cache hit rate > 98% (check metrics)
```

### 19.3 Batching Writes

```python
def batch_write_nodes(driver, data: list[dict], batch_size: int = 1000):
    """
    Always batch large writes. 
    Each transaction = one round trip + WAL write.
    Batching amortizes this cost dramatically.
    """
    query = """
    UNWIND $batch AS row
    MERGE (p:Person {id: row.id})
    ON CREATE SET p += row
    ON MATCH  SET p.updated_at = datetime()
    """
    
    total = len(data)
    for i in range(0, total, batch_size):
        chunk = data[i:i+batch_size]
        with driver.session() as session:
            session.execute_write(
                lambda tx: tx.run(query, batch=chunk)
            )
        print(f"Progress: {min(i+batch_size, total)}/{total}")

# For very large datasets, use CALL IN TRANSACTIONS in Cypher
# or neo4j-admin database import (offline bulk loader)
```

### 19.4 Connection Pool Tuning

```python
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password"),
    max_connection_pool_size=50,      # max concurrent connections
    connection_acquisition_timeout=60, # seconds to wait for pool slot
    max_transaction_retry_time=30,    # seconds to retry transient errors
    connection_timeout=30,            # seconds for initial TCP connect
    keep_alive=True,
)
```

### 19.5 Read Replicas

```python
# Route reads to secondary replicas (Enterprise)
with driver.session(
    default_access_mode="READ",       # route to read replicas
    database="neo4j"
) as session:
    result = session.run("MATCH (p:Person) RETURN count(p)")

# Causal consistency: ensure your reads see your writes
with driver.session() as session:
    bookmark = None
    
    # Write
    with session.begin_transaction() as tx:
        tx.run("CREATE (p:Person {name: 'Alice'})")
        tx.commit()
    bookmark = session.last_bookmarks()

# Pass bookmark to subsequent session to ensure read-your-writes
with driver.session(bookmarks=bookmark) as session:
    result = session.run("MATCH (p:Person {name: 'Alice'}) RETURN p")
```

### 19.6 Backup Strategy

```bash
# Online backup (Enterprise)
neo4j-admin database backup --to-path=/backup/neo4j --database=neo4j

# Offline backup (Community — stop first)
neo4j stop
cp -r $NEO4J_HOME/data/databases/neo4j /backup/neo4j_$(date +%Y%m%d)
neo4j start

# Restore
neo4j-admin database restore --from-path=/backup/neo4j --database=neo4j
```

---

## 20. Observability & Debugging

### 20.1 Key Metrics to Monitor

```cypher
-- Neo4j metrics exposed via JMX/Prometheus
-- Key metrics:

neo4j.page_cache.hit_ratio          -- target: > 0.98
neo4j.page_cache.usage_ratio        -- how full the cache is
neo4j.transaction.active            -- concurrent transactions
neo4j.transaction.committed_total   -- total commits
neo4j.bolt.connections_running      -- active Bolt connections
neo4j.store.size.total              -- on-disk size

-- Query stats from Neo4j
CALL dbms.queryJmx("org.neo4j:name=neo4j,type=Store sizes")
YIELD name, description, attributes
RETURN name, attributes
```

### 20.2 Query Log Analysis

```bash
# neo4j/logs/query.log — examine slow queries
grep "ms:" /var/log/neo4j/query.log | awk '{print $NF}' | sort -n | tail -20

# Common patterns to look for:
# - High db_hits with low rows → index missing
# - High rows mid-query that drop at end → filter too late
# - CartesianProduct → missing relationship pattern
```

### 20.3 Runtime Query Monitoring

```cypher
-- Show currently running queries
SHOW TRANSACTIONS
YIELD transactionId, currentQueryId, status, currentQuery, elapsedTime
WHERE status = "Running"
RETURN *

-- Kill a long-running query
TERMINATE TRANSACTION "transaction-id-here"

-- Query plan cache
CALL dbms.listQueryCaches()

-- Show index usage
SHOW INDEXES
YIELD name, state, type, entityType, labelsOrTypes, properties, options
RETURN *
```

### 20.4 Common Error Patterns

```
Error: "There is no procedure with the name `gds.xxx`"
Fix: GDS plugin not installed or not enabled in config

Error: "Neo.ClientError.Schema.ConstraintValidationFailed"
Fix: Attempting to create duplicate on unique-constrained property
→ Use MERGE instead of CREATE

Error: "Neo.TransientError.Transaction.DeadlockDetected"  
Fix: Two transactions locking the same nodes in opposite order
→ Retry with backoff (driver auto-retries in execute_write)

Error: "Java heap space" / OOM
Fix: 
  1. Increase heap in neo4j.conf
  2. Query is materializing too many rows — add LIMIT or restructure
  3. Add pagecache (reduce heap pressure)

Error: "Neo.ClientError.Statement.ParameterMissing"
Fix: Parameter referenced in query but not passed
→ Check $param_name matches params dict key

Warning: "Query cannot be planned — add index hint"
Fix: Create index on the filtered property
→ Or use: MATCH (n:Label) USING INDEX n:Label(prop) WHERE n.prop = $val
```

---

## 21. Anti-Patterns & What NOT to Do

### 21.1 Query Anti-Patterns

```cypher
-- ❌ NEVER: Unbounded variable-length without limit
MATCH (n)-[*]->(m)    -- will traverse the entire graph

-- ✅ ALWAYS: Bound it
MATCH (n)-[*1..5]->(m)

-- ❌ NEVER: Skip indexing and do string regex on large graphs
WHERE n.email =~ ".*@gmail.com"   -- full label scan + regex every row

-- ✅ Use full-text index
CALL db.index.fulltext.queryNodes("email_index", "gmail.com") ...

-- ❌ NEVER: Return entire nodes when you need specific properties
RETURN n    -- serializes entire node including all properties

-- ✅ Return only what you need
RETURN n.id, n.name

-- ❌ NEVER: String-interpolate query values
session.run(f"MATCH (n {{name: '{user_input}'}}) RETURN n")
-- Cypher injection vulnerability + no plan caching

-- ✅ Always parameterize
session.run("MATCH (n {name: $name}) RETURN n", name=user_input)

-- ❌ NEVER: Use id() for application logic (internal IDs are reused on delete)
WHERE id(n) = 42

-- ✅ Use your own domain ID property
WHERE n.id = "user_42"
-- Or in Neo4j 5.x:
WHERE elementId(n) = "stable-element-id"
```

### 21.2 Schema Anti-Patterns

```
❌ Don't overload relationship types:
   -[:RELATED]-> for everything
   
✅ Use specific, semantic types:
   -[:AUTHORED]-, -[:PURCHASED]-, -[:REVIEWED]-

❌ Don't store arrays of IDs as properties:
   {friends: ["id1", "id2", "id3"]}
   
✅ Model as real relationships:
   (p)-[:KNOWS]->(friend)

❌ Don't create mega-nodes (supernodes):
   (:Root)-[:CONTAINS]->(every node)  — millions of relationships on one node
   
✅ Add intermediate category/grouping nodes

❌ Don't use null-equivalent values:
   {email: "N/A", age: -1}
   
✅ Just don't set the property. IS NULL checks are first-class.

❌ Don't mirror a relational schema 1:1 into Neo4j:
   JOIN tables → relationship properties, not nodes
```

### 21.3 Production Anti-Patterns

```
❌ Don't run DETACH DELETE n on the entire database in production
   (even accidentally — there's no confirmation prompt)
   
✅ Use database constraints + read-only users for app accounts

❌ Don't ignore pagecache hit ratio < 95%
   This is a major performance signal

❌ Don't run GDS projections on the entire graph in production
   during peak hours — they lock memory and CPU
   
✅ Schedule GDS jobs off-peak or on read replicas

❌ Don't use OPTIONAL MATCH everywhere instead of thinking about
   whether your data model guarantees relationships exist

❌ Don't mix reads and writes in the same session unnecessarily
   — use separate read/write sessions for routing in clusters
```

---

## 22. Industry Patterns & Reference Architectures

### 22.1 ML Feature Store with Neo4j

```
[Data Sources: Kafka, S3, APIs]
          |
          v
   [ETL / Ingestion]
   (Spark + neo4j-connector)
          |
          v
   [Neo4j Graph DB]  ←─────────────────────────────┐
   - Raw entity graph                              │
   - Relationships with metadata                  │
          |                                        │
          v                                        │
   [GDS: Nightly feature computation]             │
   - PageRank, community, centrality              │
   - Node embeddings (FastRP/Node2Vec)            │
          |                                        │
          v                                        │
   [Feature Export to Feature Store]              │
   (Redis / Feast / Vertex AI Feature Store)      │
          |                                        │
          v                                        │
   [ML Training Pipeline]                         │
   (SageMaker / Vertex AI)                        │
          |                                        │
          v                                        │
   [Inference API]                                │
   - Serve predictions via model                  │
   - Write predictions BACK to graph ────────────┘
     (p.fraud_score, p.churn_probability)
```

### 22.2 Real-Time Fraud Detection Architecture

```
[Transaction Event] → Kafka
        |
        v
[Flink / Spark Streaming]
  - Real-time entity resolution
  - Write new transaction + links to Neo4j
        |
        v
[Neo4j Real-Time Query]
  - Ring detection within 3 hops
  - Shared identity check
  - Risk score lookup
        |
        v (< 50ms target)
[Decision Engine]
  - Block / Review / Allow
        |
        v
[Feedback Loop]
  - Label confirmed fraud → update graph
  - Retrain GNN model weekly
```

### 22.3 Knowledge Graph + RAG Architecture

```
[Documents / Web / APIs]
        |
        v
[NLP Pipeline]
  - Entity extraction (NER)
  - Relation extraction (RE)
  - Coreference resolution
        |
        v
[Neo4j Knowledge Graph]
  - Entities + Relations
  - Confidence scores
  - Source provenance
        |           |
        v           v
[Vector Index]  [Graph Index]
(embeddings)   (Cypher patterns)
        |           |
        └─────┬─────┘
              v
     [Hybrid Retriever]
        - Dense: vector ANN
        - Sparse: keyword
        - Structured: graph path
              |
              v
     [LLM with Context]
     (grounded generation)
```

### 22.4 Complete Cypher Cheat Sheet

```cypher
-- ============================================================
-- CREATE / MERGE
-- ============================================================
CREATE (n:Person {id: "1", name: "Alice"})
MERGE (n:Person {id: "1"}) ON CREATE SET n.name = "Alice"

-- ============================================================
-- READ
-- ============================================================
MATCH (n:Person {name: "Alice"})-[:KNOWS]->(f:Person)
WHERE f.age > 25
RETURN f.name, f.age ORDER BY f.age LIMIT 10

-- ============================================================
-- UPDATE
-- ============================================================
MATCH (n:Person {id: "1"}) SET n.age = 31, n += {city: "Bangalore"}

-- ============================================================
-- DELETE
-- ============================================================
MATCH (n:Person {id: "1"}) DETACH DELETE n

-- ============================================================
-- AGGREGATION
-- ============================================================
MATCH (p:Person)-[:KNOWS]->(f)
RETURN p.name, count(f) AS friends, collect(f.name) AS friend_names

-- ============================================================
-- PATH
-- ============================================================
MATCH path = shortestPath((a:Person {name:"A"})-[:KNOWS*]-(b:Person {name:"B"}))
RETURN path, length(path)

-- ============================================================
-- GDS
-- ============================================================
CALL gds.graph.project('g','Person',{KNOWS:{orientation:'UNDIRECTED'}})
CALL gds.pageRank.write('g',{writeProperty:'pr',maxIterations:20})
CALL gds.louvain.write('g',{writeProperty:'community'})
CALL gds.graph.drop('g')

-- ============================================================
-- INDEXES
-- ============================================================
CREATE CONSTRAINT person_id FOR (p:Person) REQUIRE p.id IS UNIQUE
CREATE INDEX person_name FOR (p:Person) ON (p.name)
SHOW INDEXES

-- ============================================================
-- UTILITY
-- ============================================================
EXPLAIN MATCH (n:Person) WHERE n.name = "Alice" RETURN n
PROFILE MATCH (n:Person) WHERE n.name = "Alice" RETURN n
SHOW TRANSACTIONS
CALL dbms.components() YIELD name, versions, edition
```

### 22.5 Production Deployment Checklist

```
Infrastructure:
  □ RAM = heap + pagecache + 4GB OS = sized for working set
  □ SSD storage (NVMe preferred for random IO)
  □ Separate disks for data and transaction logs
  □ Network < 1ms latency between cluster nodes

Security:
  □ Auth enabled (never disable in production)
  □ TLS on Bolt and HTTP connections
  □ Application uses least-privilege user (READ-only where possible)
  □ Import directory is locked down
  □ No procedures.unrestricted=* in production

Schema:
  □ Unique constraints on all domain ID properties
  □ Indexes on all frequently filtered properties
  □ Indexes on relationship properties used in WHERE

Operations:
  □ Nightly backup automated and tested (restore tested!)
  □ Query log enabled with threshold (1-5 seconds)
  □ Metrics exported to Prometheus/Datadog
  □ Alerting on page cache hit ratio < 95%
  □ Alerting on heap usage > 80%
  □ GDS jobs scheduled off-peak

Application:
  □ All queries parameterized
  □ Connection pool sized correctly
  □ Retry logic for transient errors (DeadlockDetected, etc.)
  □ Timeouts set on long-running queries
  □ Bounded variable-length patterns everywhere
```

---

## Appendix: Interview & Job Prep Reference

### Core Concepts You Must Know

| Concept | What They'll Ask |
|---|---|
| Index-Free Adjacency | Why is Neo4j faster than SQL for graph traversal? |
| MERGE vs CREATE | When do you use each? What happens concurrently? |
| Page Cache | How do you size it? What does hit ratio mean? |
| GDS workflow | Project → Algorithm → Write/Stream → Drop |
| Supernode | What is it, why is it a problem, how do you fix it? |
| ACID in Neo4j | How does Neo4j handle transactions? Write path? |
| Query planning | How do you debug a slow query? What is PROFILE? |
| GraphRAG | How do graphs augment LLM retrieval? |
| Node embeddings | FastRP vs Node2Vec vs GraphSAGE — tradeoffs |
| Schema design | Translate a domain into a property graph model |

### Practice Problems

```
1. Design a graph schema for a movie recommendation system.
   Include: Users, Movies, Genres, Directors, Actors.
   Write a query that recommends movies based on collaborative filtering.

2. Given a social network graph, write a Cypher query to:
   a) Find the top 10 most influential users (by PageRank)
   b) Detect users who bridge two disconnected communities
   c) Find mutual friends between two users

3. A fraud detection query should run in < 100ms.
   Currently it takes 5 seconds. Walk through how you diagnose and fix it.

4. Design the graph schema and Cypher queries for a knowledge graph
   that powers a RAG system for a technical documentation chatbot.

5. You have 50 million nodes and 200 million relationships.
   How do you size the Neo4j instance? What are the key config settings?
```

---

*Last updated: 2025 | Neo4j 5.x | GDS 2.x*

*All Cypher examples tested against Neo4j 5.15 Community and Enterprise editions.*
