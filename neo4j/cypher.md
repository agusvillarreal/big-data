# Neo4j and Cypher: A Practical Guide

```cypher
// This will delete ALL data in the current database
MATCH (n) DETACH DELETE n;
````

## Introduction to Graph Databases

Graph databases are designed to work with highly connected data where relationships are as important as the data itself. While traditional databases excel at tabular data, graph databases shine when dealing with networks of information.

Neo4j is the leading graph database platform that allows you to model, store, and query connected data efficiently. At its core, Neo4j uses a property graph model with these key components:

- **Nodes** represent entities (like people, products, events)
- **Relationships** connect nodes and represent how entities relate to each other
- **Properties** are attributes stored on both nodes and relationships

## Getting Started with Cypher

Cypher is Neo4j's graph query language designed to be visually intuitive. It uses ASCII art to represent patterns in graphs.

### Creating Your First Graph Database

Let's build a small social network step by step. Each statement below can be run in the Neo4j Browser to see immediate results.

#### Step 1: Create Person Nodes

First, let's create the nodes representing people:

```cypher
// Create Alice
CREATE (alice:Person {name: 'Alice', age: 32, occupation: 'Engineer'})
RETURN alice;

// Create Bob
CREATE (bob:Person {name: 'Bob', age: 40, occupation: 'Product Manager'})
RETURN bob;

// Create Charlie
CREATE (charlie:Person {name: 'Charlie', age: 25, occupation: 'Developer'})
RETURN charlie;

// Create David
CREATE (david:Person {name: 'David', age: 35, occupation: 'Designer'})
RETURN david;
```

After running each command, Neo4j Browser will visualize the node with its properties. You'll see a circle representing the person with their properties displayed.

#### Step 2: View All Person Nodes

Let's check all the nodes we've created:

```cypher
// Find all people in our database
MATCH (p:Person)
RETURN p;
```

This query will display all four Person nodes in the database, visualized as circles with their properties.

#### Step 3: Create Relationships Between People

Now, let's connect these people with relationships:

```cypher
// Alice KNOWS Bob
MATCH (alice:Person {name: 'Alice'}), (bob:Person {name: 'Bob'})
CREATE (alice)-[r:KNOWS {since: 2018}]->(bob)
RETURN alice, r, bob;

// Alice WORKS_WITH Charlie
MATCH (alice:Person {name: 'Alice'}), (charlie:Person {name: 'Charlie'})
CREATE (alice)-[r:WORKS_WITH {project: 'Database Migration'}]->(charlie)
RETURN alice, r, charlie;

// Bob MANAGES David
MATCH (bob:Person {name: 'Bob'}), (david:Person {name: 'David'})
CREATE (bob)-[r:MANAGES]->(david)
RETURN bob, r, david;

// Charlie KNOWS David
MATCH (charlie:Person {name: 'Charlie'}), (david:Person {name: 'David'})
CREATE (charlie)-[r:KNOWS {since: 2020}]->(david)
RETURN charlie, r, david;
```

Each query returns a visualization of the nodes connected by the newly created relationship. The relationships are displayed as arrows with their type (KNOWS, WORKS_WITH, MANAGES) labeled on them.

#### Step 4: View the Entire Social Network

Now let's see the complete social network we've created:

```cypher
// View the entire graph
MATCH (p1:Person)-[r]->(p2:Person)
RETURN p1, r, p2;
```

This will display a visualization of all nodes and their relationships in our social network, showing how everyone is connected.

### Querying Your Graph Database

Now that we have data in our graph, let's explore different ways to query it. Each of these queries will produce visualizations in the Neo4j Browser.

#### Looking at Properties

First, let's get a tabular view of all people:

```cypher
// Find all people and their details
MATCH (p:Person)
RETURN p.name, p.age, p.occupation
ORDER BY p.age;
```

This returns a table with names, ages, and occupations of all people, sorted by age.

#### Exploring Relationships

Let's find out who Alice knows:

```cypher
// Find who Alice knows
MATCH (alice:Person {name: 'Alice'})-[r:KNOWS]->(person)
RETURN alice, r, person;
```

This query produces a visualization showing Alice connected to the people she knows through KNOWS relationships.

#### Understanding Relationship Types

Let's see all the different types of relationships in our network:

```cypher
// Find all relationship types between people
MATCH (p1:Person)-[r]->(p2:Person)
RETURN p1.name AS Person1, 
       type(r) AS Relationship, 
       r AS RelationshipDetails,
       p2.name AS Person2;
```

This query returns both a visual graph and a table showing who is connected to whom and how they're connected.

#### Finding People by Relationship Properties

Let's find relationships established in 2020:

```cypher
// Find relationships from 2020
MATCH (p1:Person)-[r {since: 2020}]->(p2:Person)
RETURN p1, r, p2;
```

This query will visualize all connections that were established in 2020.

### Path Finding and Graph Traversal

One of the greatest strengths of graph databases is their ability to find paths between entities.

#### Finding All Paths Between Two People

Let's find all possible ways Alice is connected to David:

```cypher
// Find all paths between Alice and David (maximum 3 hops)
MATCH path = (alice:Person {name: 'Alice'})-[*1..3]->(david:Person {name: 'David'})
RETURN path;
```

This query finds and visualizes all paths from Alice to David that are between 1 and 3 relationships long. The visualization will show the complete paths, including any intermediate people.

#### Finding the Shortest Path

If we're only interested in the shortest connection:

```cypher
// Find shortest path between Alice and David
MATCH path = shortestPath((alice:Person {name: 'Alice'})-[*]-(david:Person {name: 'David'}))
RETURN path;
```

This will display the shortest connection between Alice and David, regardless of relationship direction.

#### Exploring Network Depth

Let's see everyone within two connections of Alice:

```cypher
// Find everyone within 2 hops of Alice
MATCH path = (alice:Person {name: 'Alice'})-[*1..2]-(person:Person)
WHERE person <> alice
RETURN path;
```

This query visualizes Alice's network up to 2 relationships away, showing how far her connections extend.

## Medium Complexity Example: Movie Recommendation System

Let's build a more complex example - a movie recommendation system that models people, movies, genres, and directors. We'll build this step by step to see how the graph grows.

### Creating the Movie Database Schema

First, let's create our movie nodes:

```cypher
// Create movie nodes one by one to see them appear
CREATE (matrix:Movie {title: 'The Matrix', released: 1999, rating: 8.7})
RETURN matrix;

CREATE (cloud:Movie {title: 'Cloud Atlas', released: 2012, rating: 7.4})
RETURN cloud;

CREATE (forrest:Movie {title: 'Forrest Gump', released: 1994, rating: 8.8})
RETURN forrest;

CREATE (inception:Movie {title: 'Inception', released: 2010, rating: 8.8})
RETURN inception;

// View all movies
MATCH (m:Movie)
RETURN m;
```

Now, let's add the people (actors):

```cypher
// Create person nodes
CREATE (keanu:Person {name: 'Keanu Reeves'})
RETURN keanu;

CREATE (tom:Person {name: 'Tom Hanks'})
RETURN tom;

CREATE (leo:Person {name: 'Leonardo DiCaprio'})
RETURN leo;

// View all actors
MATCH (p:Person)
RETURN p;
```

Next, let's add directors:

```cypher
// Create director nodes
CREATE (wachowskis:Director {name: 'The Wachowskis'})
RETURN wachowskis;

CREATE (zemeckis:Director {name: 'Robert Zemeckis'})
RETURN zemeckis;

CREATE (nolan:Director {name: 'Christopher Nolan'})
RETURN nolan;

// View all directors
MATCH (d:Director)
RETURN d;
```

Finally, let's add genres:

```cypher
// Create genre nodes
CREATE (scifi:Genre {name: 'Science Fiction'})
RETURN scifi;

CREATE (drama:Genre {name: 'Drama'})
RETURN drama;

CREATE (action:Genre {name: 'Action'})
RETURN action;

// View all genres
MATCH (g:Genre)
RETURN g;
```

### Connecting the Nodes with Relationships

Now let's connect everything together, starting with The Matrix:

```cypher
// Connect The Matrix with its actor, director and genres
MATCH (matrix:Movie {title: 'The Matrix'}), (keanu:Person {name: 'Keanu Reeves'})
CREATE (keanu)-[r:ACTED_IN {role: 'Neo'}]->(matrix)
RETURN keanu, r, matrix;

MATCH (matrix:Movie {title: 'The Matrix'}), (wachowskis:Director {name: 'The Wachowskis'})
CREATE (wachowskis)-[r:DIRECTED]->(matrix)
RETURN wachowskis, r, matrix;

MATCH (matrix:Movie {title: 'The Matrix'}), (scifi:Genre {name: 'Science Fiction'})
CREATE (matrix)-[r:IN_GENRE]->(scifi)
RETURN matrix, r, scifi;

MATCH (matrix:Movie {title: 'The Matrix'}), (action:Genre {name: 'Action'})
CREATE (matrix)-[r:IN_GENRE]->(action)
RETURN matrix, r, action;

// View The Matrix with all its relationships
MATCH (m:Movie {title: 'The Matrix'})-[r]-(n)
RETURN m, r, n;
```

Let's connect Cloud Atlas:

```cypher
// Connect Cloud Atlas with its actor, director and genres
MATCH (cloud:Movie {title: 'Cloud Atlas'}), (tom:Person {name: 'Tom Hanks'})
CREATE (tom)-[r:ACTED_IN {role: 'Multiple Characters'}]->(cloud)
RETURN tom, r, cloud;

MATCH (cloud:Movie {title: 'Cloud Atlas'}), (wachowskis:Director {name: 'The Wachowskis'})
CREATE (wachowskis)-[r:DIRECTED]->(cloud)
RETURN wachowskis, r, cloud;

MATCH (cloud:Movie {title: 'Cloud Atlas'}), (scifi:Genre {name: 'Science Fiction'})
CREATE (cloud)-[r:IN_GENRE]->(scifi)
RETURN cloud, r, scifi;

MATCH (cloud:Movie {title: 'Cloud Atlas'}), (drama:Genre {name: 'Drama'})
CREATE (cloud)-[r:IN_GENRE]->(drama)
RETURN cloud, r, drama;

// View Cloud Atlas with all its relationships
MATCH (m:Movie {title: 'Cloud Atlas'})-[r]-(n)
RETURN m, r, n;
```

Now let's connect Forrest Gump:

```cypher
// Connect Forrest Gump with its actor, director and genre
MATCH (forrest:Movie {title: 'Forrest Gump'}), (tom:Person {name: 'Tom Hanks'})
CREATE (tom)-[r:ACTED_IN {role: 'Forrest Gump'}]->(forrest)
RETURN tom, r, forrest;

MATCH (forrest:Movie {title: 'Forrest Gump'}), (zemeckis:Director {name: 'Robert Zemeckis'})
CREATE (zemeckis)-[r:DIRECTED]->(forrest)
RETURN zemeckis, r, forrest;

MATCH (forrest:Movie {title: 'Forrest Gump'}), (drama:Genre {name: 'Drama'})
CREATE (forrest)-[r:IN_GENRE]->(drama)
RETURN forrest, r, drama;

// View Forrest Gump with all its relationships
MATCH (m:Movie {title: 'Forrest Gump'})-[r]-(n)
RETURN m, r, n;
```

Finally, let's connect Inception:

```cypher
// Connect Inception with its actor, director and genres
MATCH (inception:Movie {title: 'Inception'}), (leo:Person {name: 'Leonardo DiCaprio'})
CREATE (leo)-[r:ACTED_IN {role: 'Cobb'}]->(inception)
RETURN leo, r, inception;

MATCH (inception:Movie {title: 'Inception'}), (nolan:Director {name: 'Christopher Nolan'})
CREATE (nolan)-[r:DIRECTED]->(inception)
RETURN nolan, r, inception;

MATCH (inception:Movie {title: 'Inception'}), (scifi:Genre {name: 'Science Fiction'})
CREATE (inception)-[r:IN_GENRE]->(scifi)
RETURN inception, r, scifi;

MATCH (inception:Movie {title: 'Inception'}), (action:Genre {name: 'Action'})
CREATE (inception)-[r:IN_GENRE]->(action)
RETURN inception, r, action;

// View Inception with all its relationships
MATCH (m:Movie {title: 'Inception'})-[r]-(n)
RETURN m, r, n;
```

### Viewing the Complete Movie Database

Now we can see our entire movie database:

```cypher
// View all nodes and relationships in the movie database
MATCH (n)-[r]-(m)
RETURN n, r, m;
```

This visualization shows the complete graph with all movies, people, directors, and genres connected together.

Now let's use this data model for some interesting queries:

### Exploring the Movie Database

Now that we have our movie database built, let's run some interesting queries to explore the data.

#### Finding Movies by Actor and Genre

Let's find all sci-fi movies that Tom Hanks acted in:

```cypher
// Find all sci-fi movies that Tom Hanks acted in
MATCH (tom:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(movie)-[:IN_GENRE]->(genre:Genre {name: 'Science Fiction'})
RETURN tom, movie, genre;
```

This query visualizes Tom Hanks, the sci-fi movies he's acted in, and the Science Fiction genre node all connected together.

We can also get just the movie details in a table format:

```cypher
// Table of sci-fi movies with Tom Hanks
MATCH (tom:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(movie)-[:IN_GENRE]->(genre:Genre {name: 'Science Fiction'})
RETURN movie.title, movie.released, movie.rating
ORDER BY movie.rating DESC;
```

#### Finding All Movies by Genre

Let's see all movies in each genre:

```cypher
// Find all movies by genre
MATCH (movie:Movie)-[:IN_GENRE]->(genre)
RETURN genre.name AS Genre, collect(movie.title) AS Movies, count(movie) AS MovieCount
ORDER BY MovieCount DESC;
```

This returns a table grouping movies by genre with counts.

#### Building a Movie Recommendation Engine

Let's build a simple recommendation engine that finds movies similar to ones a user has watched:

```cypher
// Recommend movies based on shared genres with "The Matrix"
MATCH (matrix:Movie {title: 'The Matrix'})-[:IN_GENRE]->(genre)<-[:IN_GENRE]-(recommendation)
WHERE recommendation <> matrix
RETURN matrix, genre, recommendation;
```

This visualizes The Matrix, its genres, and other movies that share those genres.

For a more sortable table of recommendations:

```cypher
// Rank recommendations by genre overlap and rating
MATCH (matrix:Movie {title: 'The Matrix'})-[:IN_GENRE]->(genre)<-[:IN_GENRE]-(recommendation)
WHERE recommendation <> matrix
RETURN recommendation.title AS RecommendedMovie, 
       recommendation.rating AS Rating, 
       collect(genre.name) AS SharedGenres,
       count(genre) AS GenreOverlap
ORDER BY GenreOverlap DESC, Rating DESC;
```

This query:

1. Starts with "The Matrix"
2. Finds all genres it belongs to
3. Finds other movies in those same genres
4. Ranks recommendations by genre overlap and then by rating

#### Finding Actors Who Worked with the Same Director

Let's find all actors who worked with Christopher Nolan:

```cypher
// Find all actors who worked with Christopher Nolan
MATCH (actor:Person)-[:ACTED_IN]->(movie)<-[:DIRECTED]-(director:Director {name: 'Christopher Nolan'})
RETURN director, movie, actor;
```

This visualizes Christopher Nolan, his movies, and the actors who appeared in them.

#### Advanced Path Analysis - Finding Connections Between Actors

Let's find how actors are connected through movies:

```cypher
// Find how Keanu Reeves and Leonardo DiCaprio are connected
MATCH path = shortestPath((keanu:Person {name: 'Keanu Reeves'})-[:ACTED_IN|:DIRECTED*..10]-(leo:Person {name: 'Leonardo DiCaprio'}))
RETURN path;
```

This finds and visualizes the shortest connection between Keanu Reeves and Leonardo DiCaprio through the movie database.

For a more readable output of the connection:

```cypher
// Show the connection path in readable format
MATCH path = shortestPath((keanu:Person {name: 'Keanu Reeves'})-[:ACTED_IN|:DIRECTED*..10]-(leo:Person {name: 'Leonardo DiCaprio'}))
RETURN [node in nodes(path) | 
  CASE 
    WHEN node:Person THEN node.name + ' (Actor)'
    WHEN node:Director THEN node.name + ' (Director)'
    WHEN node:Movie THEN node.title + ' (Movie)'
    ELSE 'Unknown'
  END] AS ConnectionPath;
```

This returns the nodes in the path with labels to understand the connection.

## Neo4j vs. Traditional SQL Databases

### Data Modeling Differences

**SQL Database Approach:**

In a relational database, our movie example would require multiple tables:

```sql
CREATE TABLE People (
    person_id INT PRIMARY KEY,
    name VARCHAR(100)
);

CREATE TABLE Movies (
    movie_id INT PRIMARY KEY,
    title VARCHAR(100),
    released INT,
    rating DECIMAL(3,1)
);

CREATE TABLE Directors (
    director_id INT PRIMARY KEY,
    name VARCHAR(100)
);

CREATE TABLE Genres (
    genre_id INT PRIMARY KEY,
    name VARCHAR(50)
);

CREATE TABLE ActedIn (
    person_id INT,
    movie_id INT,
    role VARCHAR(100),
    PRIMARY KEY (person_id, movie_id),
    FOREIGN KEY (person_id) REFERENCES People(person_id),
    FOREIGN KEY (movie_id) REFERENCES Movies(movie_id)
);

CREATE TABLE Directed (
    director_id INT,
    movie_id INT,
    PRIMARY KEY (director_id, movie_id),
    FOREIGN KEY (director_id) REFERENCES Directors(director_id),
    FOREIGN KEY (movie_id) REFERENCES Movies(movie_id)
);

CREATE TABLE MovieGenres (
    movie_id INT,
    genre_id INT,
    PRIMARY KEY (movie_id, genre_id),
    FOREIGN KEY (movie_id) REFERENCES Movies(movie_id),
    FOREIGN KEY (genre_id) REFERENCES Genres(genre_id)
);
```

**Key Differences:**

1. **Schema Rigidity**:
    
    - SQL databases require predefined schemas with tables, columns, and relationships established upfront
    - Neo4j is schema-optional, allowing you to add new node types, relationships, and properties without migrations
2. **Relationship Handling**:
    
    - SQL databases represent relationships through foreign keys and junction tables
    - Neo4j makes relationships first-class citizens with their own properties
3. **Query Complexity for Traversals**:
    
    - Finding paths in SQL requires multiple joins that grow exponentially with path length
    - In Neo4j, traversals are natural and performant regardless of depth

### Query Approach Comparison

Let's compare how we'd query "actors who appeared in sci-fi movies" in both systems:

**SQL Query:**

```sql
SELECT p.name, m.title
FROM People p
JOIN ActedIn a ON p.person_id = a.person_id
JOIN Movies m ON a.movie_id = m.movie_id
JOIN MovieGenres mg ON m.movie_id = mg.movie_id
JOIN Genres g ON mg.genre_id = g.genre_id
WHERE g.name = 'Science Fiction';
```

**Cypher Query:**

```cypher
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)-[:IN_GENRE]->(g:Genre {name: 'Science Fiction'})
RETURN p.name, m.title;
```

The Neo4j query is not only shorter but visually represents the pattern we're looking for.

### Performance Characteristics

**SQL Databases Excel At:**

- Simple, predictable queries on structured data
- Transactions requiring ACID guarantees
- Aggregate operations across large datasets
- Well-understood scaling patterns

**Neo4j Excels At:**

- Traversing deep relationships (friend-of-friend queries)
- Finding shortest paths between entities
- Detecting patterns within connected data
- Maintaining relationship integrity without expensive joins

### Use Case Suitability

Neo4j is particularly well-suited for:

1. **Social Networks**: Modeling connections between people, posts, interests
    
2. **Recommendation Engines**: Finding similar products, content, or people based on multiple shared attributes
    
3. **Fraud Detection**: Identifying suspicious patterns of connections
    
4. **Knowledge Graphs**: Representing complex domains with many entity types and relationships
    
5. **Network Management**: Mapping dependencies in IT infrastructure or logistics
    
6. **Identity and Access Management**: Modeling complex permissions hierarchies
    

Traditional SQL databases remain better for:

1. **Financial Systems**: Where strict ACID compliance and reliable transactions are critical
    
2. **Business Reporting**: Where aggregated data across many records is the primary access pattern
    
3. **Systems of Record**: Where structured data with minimal relationships is the focus
    
4. **High-volume Transactional Systems**: Where throughput of simple operations is prioritized
    

## Conclusion

Neo4j and graph databases provide a powerful approach for working with connected data. The intuitive Cypher query language makes it accessible to express complex data relationships that would be cumbersome in SQL. While not a replacement for relational databases in all scenarios, graph databases excel at traversing relationships and finding patterns within interconnected data.

As data becomes increasingly connected in our digital world, graph databases like Neo4j provide a natural fit for many modern use cases from social networks to recommendation engines, fraud detection, and beyond.
