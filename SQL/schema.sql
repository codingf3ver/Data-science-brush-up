CREATE DATABASE IF NOT EXISTS analysis;
USE analysis;

CREATE TABLE departments (
    department_id INT PRIMARY KEY,
    department_name VARCHAR(50)
);

CREATE TABLE cities (
    city_id INT PRIMARY KEY,
    city_name VARCHAR(50),
    state VARCHAR(50),
    population INT
);

CREATE TABLE employees (
    employee_id INT PRIMARY KEY,
    name VARCHAR(50),
    department_id INT,
    salary FLOAT,
    join_date DATE,
    city_id INT,
    bonus FLOAT,
    FOREIGN KEY (department_id) REFERENCES departments(department_id),
    FOREIGN KEY (city_id) REFERENCES cities(city_id)
);

CREATE TABLE employee_performance (
    performance_id INT PRIMARY KEY,
    employee_id INT,
    review_year INT,
    rating INT,
    remarks VARCHAR(255),
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
);