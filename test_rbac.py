from supabase import create_client

SUPABASE_URL = "https://pdkcqblaflpbvtrcbxua.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBka2NxYmxhZmxwYnZ0cmNieHVhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE0OTY4MDcsImV4cCI6MjA4NzA3MjgwN30.rPHMoh5Na70m6rsg-NiSkMEZ2A6KifBt1qUJGB0Of-8"

email = "gg@gmail.com"
password = "GrishmGhosh"

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Login
response = supabase.auth.sign_in_with_password({
    "email": email,
    "password": password
})

if response.user is None:
    print("Login failed")
    exit()

print("Logged in as:", response.user.id)

# Try INSERT (should fail for member)
try:
    insert = supabase.table("transactions").insert({
        "org_id": "org_001",
        "voucher_number": "TEST123",
        "amount": 1000,
        "year": 2025,
        "month": 2
    }).execute()

    print("Insert result:", insert.data)

except Exception as e:
    print("Insert failed as expected:", e)

# Try SELECT (should succeed)
select = supabase.table("transactions").select("*").execute()
print("Select result:", select.data)