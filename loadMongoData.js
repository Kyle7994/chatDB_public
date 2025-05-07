const dbName = 'university';
db = db.getSiblingDB(dbName);

const collections = db.getCollectionNames();

function loadJSONFromFile(filename) {
    const fileContent = fs.readFileSync(filename, 'utf8');
    return JSON.parse(fileContent);
  }

// students
if (!collections.includes('students')) {
  db.createCollection('students');
  const students = loadJSONFromFile('/docker-entrypoint-initdb.d/mongo-data-files/students.json');
  db.students.insertMany(students);
}

// courses
if (!collections.includes('courses')) {
  db.createCollection('courses');
  const courses = loadJSONFromFile('/docker-entrypoint-initdb.d/mongo-data-files/courses.json');
  db.courses.insertMany(courses);
}

// enrollments
if (!collections.includes('enrollments')) {
  db.createCollection('enrollments');
  const enrollments = loadJSONFromFile('/docker-entrypoint-initdb.d/mongo-data-files/enrollments.json');
  db.enrollments.insertMany(enrollments);
}

