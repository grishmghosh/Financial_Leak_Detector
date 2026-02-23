from supabase import create_client

SUPABASE_URL = "https://pdkcqblaflpbvtrcbxua.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBka2NxYmxhZmxwYnZ0cmNieHVhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE0OTY4MDcsImV4cCI6MjA4NzA3MjgwN30.rPHMoh5Na70m6rsg-NiSkMEZ2A6KifBt1qUJGB0Of-8"

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

response = supabase.auth.sign_up({
    "email": "badbunny@gmail.com",
    "password": "GrishmGhosh"
})

print(response)