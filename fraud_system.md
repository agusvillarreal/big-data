# Fraud Detection System with Neo4j

Fraud detection is one of the most powerful applications of graph databases. By modeling transactions, accounts, and entities as a connected network, we can identify suspicious patterns that would be difficult to detect in traditional systems.

## Building the Fraud Detection Database

Let's create a financial transaction database that we can use to detect potential fraud patterns.

### 1. Create Schema Constraints

First, let's set up some constraints to ensure uniqueness:

```cypher
// Set constraints to ensure uniqueness
CREATE CONSTRAINT IF NOT EXISTS FOR (a:Account) REQUIRE a.accountId IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (ip:IPAddress) REQUIRE ip.address IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (d:Device) REQUIRE d.deviceId IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (t:Transaction) REQUIRE t.transactionId IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (a:Address) REQUIRE a.id IS UNIQUE;
```

### 2. Create Accounts and People

Let's create some individuals and their accounts:

```cypher
// Create people
MERGE (alice:Person {id: 'P001', name: 'Alice Johnson', ssn: '123-45-6789'})
RETURN alice;

MERGE (bob:Person {id: 'P002', name: 'Bob Smith', ssn: '234-56-7890'})
RETURN bob;

MERGE (charlie:Person {id: 'P003', name: 'Charlie Brown', ssn: '345-67-8901'})
RETURN charlie;

MERGE (dave:Person {id: 'P004', name: 'Dave Wilson', ssn: '456-78-9012'})
RETURN dave;

MERGE (eve:Person {id: 'P005', name: 'Eve Davis', ssn: '567-89-0123'})
RETURN eve;

MERGE (frank:Person {id: 'P006', name: 'Frank Miller', ssn: '678-90-1234'})
RETURN frank;

// Create accounts
MERGE (account1:Account {accountId: 'A001', type: 'Checking', balance: 5000})
RETURN account1;

MERGE (account2:Account {accountId: 'A002', type: 'Savings', balance: 25000})
RETURN account2;

MERGE (account3:Account {accountId: 'A003', type: 'Checking', balance: 2500})
RETURN account3;

MERGE (account4:Account {accountId: 'A004', type: 'Investment', balance: 100000})
RETURN account4;

MERGE (account5:Account {accountId: 'A005', type: 'Checking', balance: 7500})
RETURN account5;

MERGE (account6:Account {accountId: 'A006', type: 'Savings', balance: 15000})
RETURN account6;

MERGE (account7:Account {accountId: 'A007', type: 'Checking', balance: 3000})
RETURN account7;
```

### 3. Create Relationships Between People and Accounts

Now let's connect people to their accounts:

```cypher
// Connect people to accounts
MATCH (alice:Person {id: 'P001'}), (account1:Account {accountId: 'A001'})
MERGE (alice)-[r:OWNS]->(account1)
RETURN alice, r, account1;

MATCH (alice:Person {id: 'P001'}), (account2:Account {accountId: 'A002'})
MERGE (alice)-[r:OWNS]->(account2)
RETURN alice, r, account2;

MATCH (bob:Person {id: 'P002'}), (account3:Account {accountId: 'A003'})
MERGE (bob)-[r:OWNS]->(account3)
RETURN bob, r, account3;

MATCH (charlie:Person {id: 'P003'}), (account4:Account {accountId: 'A004'})
MERGE (charlie)-[r:OWNS]->(account4)
RETURN charlie, r, account4;

MATCH (dave:Person {id: 'P004'}), (account5:Account {accountId: 'A005'})
MERGE (dave)-[r:OWNS]->(account5)
RETURN dave, r, account5;

MATCH (eve:Person {id: 'P005'}), (account6:Account {accountId: 'A006'})
MERGE (eve)-[r:OWNS]->(account6)
RETURN eve, r, account6;

MATCH (frank:Person {id: 'P006'}), (account7:Account {accountId: 'A007'})
MERGE (frank)-[r:OWNS]->(account7)
RETURN frank, r, account7;
```

### 4. Add Physical Addresses

Let's add physical addresses for the people:

```cypher
// Create addresses
MERGE (addr1:Address {id: 'ADDR001', street: '123 Main St', city: 'Boston', state: 'MA', zip: '02108'})
RETURN addr1;

MERGE (addr2:Address {id: 'ADDR002', street: '456 Oak Ave', city: 'New York', state: 'NY', zip: '10001'})
RETURN addr2;

MERGE (addr3:Address {id: 'ADDR003', street: '789 Pine Rd', city: 'Chicago', state: 'IL', zip: '60601'})
RETURN addr3;

MERGE (addr4:Address {id: 'ADDR004', street: '101 Maple Dr', city: 'San Francisco', state: 'CA', zip: '94101'})
RETURN addr4;

// Connect people to addresses
MATCH (alice:Person {id: 'P001'}), (addr1:Address {id: 'ADDR001'})
MERGE (alice)-[r:LIVES_AT]->(addr1)
RETURN alice, r, addr1;

MATCH (bob:Person {id: 'P002'}), (addr2:Address {id: 'ADDR002'})
MERGE (bob)-[r:LIVES_AT]->(addr2)
RETURN bob, r, addr2;

MATCH (charlie:Person {id: 'P003'}), (addr3:Address {id: 'ADDR003'})
MERGE (charlie)-[r:LIVES_AT]->(addr3)
RETURN charlie, r, addr3;

MATCH (dave:Person {id: 'P004'}), (addr4:Address {id: 'ADDR004'})
MERGE (dave)-[r:LIVES_AT]->(addr4)
RETURN dave, r, addr4;

// Fraudsters sharing an address (red flag)
MATCH (eve:Person {id: 'P005'}), (addr2:Address {id: 'ADDR002'})
MERGE (eve)-[r:LIVES_AT]->(addr2)
RETURN eve, r, addr2;

MATCH (frank:Person {id: 'P006'}), (addr2:Address {id: 'ADDR002'})
MERGE (frank)-[r:LIVES_AT]->(addr2)
RETURN frank, r, addr2;
```

### 5. Add Digital Footprints (IPs and Devices)

Let's add digital identifiers:

```cypher
// Create IP addresses
MERGE (ip1:IPAddress {address: '192.168.1.1'})
RETURN ip1;

MERGE (ip2:IPAddress {address: '192.168.1.2'})
RETURN ip2;

MERGE (ip3:IPAddress {address: '192.168.1.3'})
RETURN ip3;

MERGE (ip4:IPAddress {address: '192.168.1.4'})
RETURN ip4;

// Create devices
MERGE (device1:Device {deviceId: 'D001', type: 'iPhone', fingerprint: 'abcdef123456'})
RETURN device1;

MERGE (device2:Device {deviceId: 'D002', type: 'Android', fingerprint: 'bcdefg234567'})
RETURN device2;

MERGE (device3:Device {deviceId: 'D003', type: 'Windows PC', fingerprint: 'cdefgh345678'})
RETURN device3;

MERGE (device4:Device {deviceId: 'D004', type: 'MacBook', fingerprint: 'defghi456789'})
RETURN device4;

// Connect people to digital footprints
MATCH (alice:Person {id: 'P001'}), (ip1:IPAddress {address: '192.168.1.1'})
MERGE (alice)-[r:ACCESSED_FROM]->(ip1)
RETURN alice, r, ip1;

MATCH (bob:Person {id: 'P002'}), (ip2:IPAddress {address: '192.168.1.2'})
MERGE (bob)-[r:ACCESSED_FROM]->(ip2)
RETURN bob, r, ip2;

MATCH (charlie:Person {id: 'P003'}), (ip3:IPAddress {address: '192.168.1.3'})
MERGE (charlie)-[r:ACCESSED_FROM]->(ip3)
RETURN charlie, r, ip3;

MATCH (dave:Person {id: 'P004'}), (ip4:IPAddress {address: '192.168.1.4'})
MERGE (dave)-[r:ACCESSED_FROM]->(ip4)
RETURN dave, r, ip4;

// Suspicious shared IP (red flag)
MATCH (eve:Person {id: 'P005'}), (ip2:IPAddress {address: '192.168.1.2'})
MERGE (eve)-[r:ACCESSED_FROM]->(ip2)
RETURN eve, r, ip2;

MATCH (alice:Person {id: 'P001'}), (device1:Device {deviceId: 'D001'})
MERGE (alice)-[r:USES]->(device1)
RETURN alice, r, device1;

MATCH (bob:Person {id: 'P002'}), (device2:Device {deviceId: 'D002'})
MERGE (bob)-[r:USES]->(device2)
RETURN bob, r, device2;

MATCH (charlie:Person {id: 'P003'}), (device3:Device {deviceId: 'D003'})
MERGE (charlie)-[r:USES]->(device3)
RETURN charlie, r, device3;

MATCH (dave:Person {id: 'P004'}), (device4:Device {deviceId: 'D004'})
MERGE (dave)-[r:USES]->(device4)
RETURN dave, r, device4;

// Suspicious shared device (red flag)
MATCH (frank:Person {id: 'P006'}), (device2:Device {deviceId: 'D002'})
MERGE (frank)-[r:USES]->(device2)
RETURN frank, r, device2;
```

### 6. Create Transactions

Now let's create some financial transactions, including some suspicious ones:

```cypher
// Normal transactions
MERGE (t1:Transaction {transactionId: 'T001', amount: 100, timestamp: datetime('2023-01-01T10:30:00'), status: 'Completed'})
RETURN t1;

MERGE (t2:Transaction {transactionId: 'T002', amount: 200, timestamp: datetime('2023-01-02T14:20:00'), status: 'Completed'})
RETURN t2;

MERGE (t3:Transaction {transactionId: 'T003', amount: 50, timestamp: datetime('2023-01-03T09:15:00'), status: 'Completed'})
RETURN t3;

MERGE (t4:Transaction {transactionId: 'T004', amount: 75, timestamp: datetime('2023-01-04T16:45:00'), status: 'Completed'})
RETURN t4;

// Suspicious transactions (red flags)
MERGE (t5:Transaction {transactionId: 'T005', amount: 9500, timestamp: datetime('2023-01-05T11:10:00'), status: 'Completed'})
RETURN t5;

MERGE (t6:Transaction {transactionId: 'T006', amount: 9000, timestamp: datetime('2023-01-05T11:20:00'), status: 'Completed'})
RETURN t6;

MERGE (t7:Transaction {transactionId: 'T007', amount: 9700, timestamp: datetime('2023-01-05T11:30:00'), status: 'Completed'})
RETURN t7;

// Connect accounts to transactions (normal ones)
MATCH (account1:Account {accountId: 'A001'}), (account3:Account {accountId: 'A003'}), (t1:Transaction {transactionId: 'T001'})
MERGE (account1)-[r1:SENT]->(t1)-[r2:RECEIVED_BY]->(account3)
RETURN account1, r1, t1, r2, account3;

MATCH (account3:Account {accountId: 'A003'}), (account5:Account {accountId: 'A005'}), (t2:Transaction {transactionId: 'T002'})
MERGE (account3)-[r1:SENT]->(t2)-[r2:RECEIVED_BY]->(account5)
RETURN account3, r1, t2, r2, account5;

MATCH (account5:Account {accountId: 'A005'}), (account1:Account {accountId: 'A001'}), (t3:Transaction {transactionId: 'T003'})
MERGE (account5)-[r1:SENT]->(t3)-[r2:RECEIVED_BY]->(account1)
RETURN account5, r1, t3, r2, account1;

MATCH (account4:Account {accountId: 'A004'}), (account2:Account {accountId: 'A002'}), (t4:Transaction {transactionId: 'T004'})
MERGE (account4)-[r1:SENT]->(t4)-[r2:RECEIVED_BY]->(account2)
RETURN account4, r1, t4, r2, account2;

// Connect accounts to suspicious transactions (structuring pattern)
MATCH (account6:Account {accountId: 'A006'}), (account7:Account {accountId: 'A007'}), (t5:Transaction {transactionId: 'T005'})
MERGE (account6)-[r1:SENT]->(t5)-[r2:RECEIVED_BY]->(account7)
RETURN account6, r1, t5, r2, account7;

MATCH (account6:Account {accountId: 'A006'}), (account7:Account {accountId: 'A007'}), (t6:Transaction {transactionId: 'T006'})
MERGE (account6)-[r1:SENT]->(t6)-[r2:RECEIVED_BY]->(account7)
RETURN account6, r1, t6, r2, account7;

MATCH (account6:Account {accountId: 'A006'}), (account7:Account {accountId: 'A007'}), (t7:Transaction {transactionId: 'T007'})
MERGE (account6)-[r1:SENT]->(t7)-[r2:RECEIVED_BY]->(account7)
RETURN account6, r1, t7, r2, account7;

// Add IP and device context to transactions
MATCH (t1:Transaction {transactionId: 'T001'}), (ip1:IPAddress {address: '192.168.1.1'}), (device1:Device {deviceId: 'D001'})
MERGE (t1)-[r1:MADE_FROM]->(ip1)
MERGE (t1)-[r2:USED_DEVICE]->(device1)
RETURN t1, r1, ip1, r2, device1;

MATCH (t2:Transaction {transactionId: 'T002'}), (ip2:IPAddress {address: '192.168.1.2'}), (device2:Device {deviceId: 'D002'})
MERGE (t2)-[r1:MADE_FROM]->(ip2)
MERGE (t2)-[r2:USED_DEVICE]->(device2)
RETURN t2, r1, ip2, r2, device2;

// Suspicious transactions from unexpected locations/devices
MATCH (t5:Transaction {transactionId: 'T005'}), (ip4:IPAddress {address: '192.168.1.4'}), (device3:Device {deviceId: 'D003'})
MERGE (t5)-[r1:MADE_FROM]->(ip4)
MERGE (t5)-[r2:USED_DEVICE]->(device3)
RETURN t5, r1, ip4, r2, device3;

MATCH (t6:Transaction {transactionId: 'T006'}), (ip4:IPAddress {address: '192.168.1.4'}), (device3:Device {deviceId: 'D003'})
MERGE (t6)-[r1:MADE_FROM]->(ip4)
MERGE (t6)-[r2:USED_DEVICE]->(device3)
RETURN t6, r1, ip4, r2, device3;

MATCH (t7:Transaction {transactionId: 'T007'}), (ip4:IPAddress {address: '192.168.1.4'}), (device3:Device {deviceId: 'D003'})
MERGE (t7)-[r1:MADE_FROM]->(ip4)
MERGE (t7)-[r2:USED_DEVICE]->(device3)
RETURN t7, r1, ip4, r2, device3;
```

### 7. View the Complete Fraud Detection Graph

Let's look at our fraud detection graph:

```cypher
// View all nodes and relationships
MATCH (n)
RETURN n;

// Alternative, more focused view of just the transaction paths
MATCH path = (sender:Account)-[:SENT]->(t:Transaction)-[:RECEIVED_BY]->(receiver:Account)
RETURN path;
```

## Fraud Detection Patterns

Now that we have our graph database populated, let's explore common fraud detection patterns.

### 1. Detect Structuring (Multiple Transactions Just Under Reporting Thresholds)

Structuring is a technique used to avoid transaction reporting requirements by breaking large transactions into multiple smaller ones, each just under the reporting threshold (typically $10,000).

```cypher
// Find multiple transactions just under $10,000 between the same accounts in a short time period
MATCH (sender:Account)-[:SENT]->(t:Transaction)-[:RECEIVED_BY]->(receiver:Account)
WHERE t.amount > 8000 AND t.amount < 10000
WITH sender, receiver, collect(t) AS transactions
WHERE size(transactions) >= 3
RETURN sender, receiver, transactions,
       sum(transaction IN transactions | transaction.amount) AS totalAmount,
       size(transactions) AS transactionCount;

// Visualize the structuring pattern
MATCH (sender:Account)-[:SENT]->(t:Transaction)-[:RECEIVED_BY]->(receiver:Account)
WHERE t.amount > 8000 AND t.amount < 10000
WITH sender, receiver, collect(t) AS transactions 
WHERE size(transactions) >= 3
MATCH (sender)-[:SENT]->(t:Transaction)-[:RECEIVED_BY]->(receiver)
WHERE t IN transactions
RETURN sender, t, receiver;
```

### 2. Detect Shared Identifiers (Red Flag for Identity Theft or Synthetic Identities)

Legitimate users typically don't share important identifiers. Multiple accounts linked to the same device, IP, or address could indicate fraud.

```cypher
// Find accounts accessed from the same IP address by different people
MATCH (p1:Person)-[:OWNS]->(a1:Account),
      (p2:Person)-[:OWNS]->(a2:Account),
      (p1)-[:ACCESSED_FROM]->(ip)<-[:ACCESSED_FROM]-(p2)
WHERE p1 <> p2
RETURN p1.name AS Person1, a1.accountId AS Account1, 
       p2.name AS Person2, a2.accountId AS Account2, 
       ip.address AS SharedIPAddress;

// Visualize people sharing the same device
MATCH (p1:Person)-[:USES]->(device)<-[:USES]-(p2:Person)
WHERE p1 <> p2
RETURN p1, device, p2;

// Visualize people sharing the same address
MATCH (p1:Person)-[:LIVES_AT]->(address)<-[:LIVES_AT]-(p2:Person)
WHERE p1 <> p2
RETURN p1, address, p2;
```

### 3. Detect Unusual Transaction Patterns

Identifying transactions that don't match a user's normal behavior:

```cypher
// Find transactions made from devices or IPs not typically used by the account owner
MATCH (p:Person)-[:OWNS]->(a:Account)-[:SENT]->(t:Transaction)-[:USED_DEVICE]->(d:Device)
WHERE NOT (p)-[:USES]->(d)
RETURN p.name AS Person, a.accountId AS Account, 
       t.transactionId AS Transaction, t.amount AS Amount, 
       d.deviceId AS UnusualDevice;

// Visualize the unusual activity path
MATCH path = (p:Person)-[:OWNS]->(a:Account)-[:SENT]->(t:Transaction)-[:USED_DEVICE]->(d:Device)
WHERE NOT (p)-[:USES]->(d)
RETURN path;
```

### 4. Detect Money Laundering Cycles

Detecting cycles in transaction graphs can reveal money laundering:

```cypher
// Find potential money laundering cycles where money flows in a circle
MATCH path = (a:Account)-[:SENT]->(:Transaction)-[:RECEIVED_BY]->(b:Account)-[:SENT]->
             (:Transaction)-[:RECEIVED_BY]->(c:Account)-[:SENT]->(:Transaction)-[:RECEIVED_BY]->(a)
RETURN path;
```

### 5. Detect First-Party Fraud Rings

First-party fraud often involves networks of synthetic or stolen identities controlled by the same fraudster:

```cypher
// Find potential fraud rings based on shared attributes
MATCH (p1:Person)-[:OWNS]->(a1:Account),
      (p2:Person)-[:OWNS]->(a2:Account)
WHERE p1 <> p2 AND 
      (
        (p1)-[:LIVES_AT]->()<-[:LIVES_AT]-(p2) OR
        (p1)-[:ACCESSED_FROM]->()<-[:ACCESSED_FROM]-(p2) OR
        (p1)-[:USES]->()<-[:USES]-(p2)
      )
WITH p1, p2, count(*) AS commonAttributes
WHERE commonAttributes >= 2
RETURN p1.name AS Person1, p2.name AS Person2, commonAttributes;

// Visualize the potential fraud ring
MATCH (p1:Person)-[r1]-(shared)-[r2]-(p2:Person)
WHERE p1 <> p2 AND type(r1) = type(r2) AND 
      (type(r1) = 'LIVES_AT' OR type(r1) = 'ACCESSED_FROM' OR type(r1) = 'USES')
WITH p1, p2, count(shared) AS commonAttributes
WHERE commonAttributes >= 2
MATCH path = shortestPath((p1)-[*..6]-(p2))
RETURN path;
```

## Advanced Fraud Detection Techniques

### 1. Paginating Through Fraud Alerts

In a real system, you might need to paginate through fraud alerts:

```cypher
// Paginated view of suspicious transactions (10 per page)
MATCH (sender:Account)-[:SENT]->(t:Transaction)-[:RECEIVED_BY]->(receiver:Account)
WHERE t.amount > 8000 AND t.amount < 10000
RETURN sender.accountId AS SenderAccount, 
       receiver.accountId AS ReceiverAccount,
       t.transactionId AS TransactionID, 
       t.amount AS Amount, 
       t.timestamp AS Timestamp
ORDER BY t.timestamp DESC
SKIP 0 LIMIT 10;
```

### 2. Real-time Fraud Scoring

Develop a fraud score based on multiple risk factors:

```cypher
// Calculate fraud risk score based on various signals
MATCH (p:Person)-[:OWNS]->(a:Account)-[:SENT]->(t:Transaction)
OPTIONAL MATCH (t)-[:USED_DEVICE]->(d:Device)
OPTIONAL MATCH (t)-[:MADE_FROM]->(ip:IPAddress)

WITH p, a, t, d, ip,
     CASE WHEN t.amount > 8000 AND t.amount < 10000 THEN 25 ELSE 0 END +
     CASE WHEN NOT (p)-[:USES]->(d) THEN 20 ELSE 0 END +
     CASE WHEN NOT (p)-[:ACCESSED_FROM]->(ip) THEN 20 ELSE 0 END AS riskScore,
     CASE 
       WHEN t.amount > 8000 AND t.amount < 10000 THEN 'Near-threshold transaction, '
       ELSE ''
     END +
     CASE 
       WHEN NOT (p)-[:USES]->(d) THEN 'Unusual device, '
       ELSE ''
     END +
     CASE 
       WHEN NOT (p)-[:ACCESSED_FROM]->(ip) THEN 'Unusual location, '
       ELSE ''
     END AS riskFactors

WHERE riskScore > 0
RETURN p.name AS PersonName, 
       a.accountId AS AccountID,
       t.transactionId AS TransactionID,
       t.amount AS Amount,
       riskScore AS FraudRiskScore,
       riskFactors AS RiskFactors
ORDER BY riskScore DESC;
```

## Conclusion

This example demonstrates how Neo4j and graph databases excel at fraud detection by:

1. Making complex relationships between entities easy to model and query
2. Enabling pattern detection across multiple degrees of separation
3. Allowing for visual investigation and analysis of suspicious patterns
4. Supporting both rules-based and machine learning approaches to fraud detection

The same patterns would be extremely difficult to implement in a traditional relational database, requiring numerous complex joins and recursive queries. In Neo4j, these relationships are first-class citizens that make fraud pattern detection both intuitive and performant.
