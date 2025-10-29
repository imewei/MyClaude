
using Genie.Router
using JSON3

route("/") do
    JSON3.write(Dict("message" => "Hello from Julia API"))
end

route("/api/users", method = GET) do
    users = [
        Dict("id" => 1, "name" => "Alice"),
        Dict("id" => 2, "name" => "Bob")
    ]
    JSON3.write(users)
end

route("/api/users", method = POST) do
    payload = jsonpayload()
    # Process user creation
    JSON3.write(Dict("status" => "created", "user" => payload))
end
