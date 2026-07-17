-- Supabase schema for the assignment grading agent

create extension if not exists "pgcrypto"; -- for gen_random_uuid()

create table assignments (
  id uuid primary key default gen_random_uuid(),
  title text not null,
  max_score numeric not null default 100,
  rubric_summary text,
  created_at timestamptz default now()
);

create table students (
  id uuid primary key default gen_random_uuid(),
  name text,
  email text unique
);

create table submissions (
  id uuid primary key default gen_random_uuid(),
  assignment_id uuid references assignments(id),
  student_id uuid references students(id),
  raw_text text,
  file_url text,
  submitted_at timestamptz default now()
);

create table evaluations (
  id uuid primary key default gen_random_uuid(),
  assignment_id uuid references assignments(id),
  student_id text,               -- free-text id/email at submit time (no login required)
  score numeric,
  feedback_json jsonb,           -- full structured Evaluation object from the agent
  flagged_for_review boolean default false,
  evaluated_at timestamptz default now()
);

-- Optional: row-level security if this becomes multi-tenant / multi-teacher
-- alter table assignments enable row level security;
-- alter table evaluations enable row level security;
