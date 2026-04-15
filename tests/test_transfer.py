import pytest
from botocore.exceptions import ClientError

# ========== Server lifecycle ==========

def test_transfer_create_server(transfer):
    resp = transfer.create_server()
    assert "ServerId" in resp
    assert resp["ServerId"].startswith("s-")


def test_transfer_describe_server(transfer):
    sid = transfer.create_server()["ServerId"]
    resp = transfer.describe_server(ServerId=sid)
    server = resp["Server"]
    assert server["ServerId"] == sid
    assert server["State"] == "ONLINE"
    assert server["EndpointType"] == "PUBLIC"
    assert server["IdentityProviderType"] == "SERVICE_MANAGED"
    assert "SFTP" in server["Protocols"]
    assert server["Arn"].startswith("arn:aws:transfer:")


def test_transfer_describe_server_not_found(transfer):
    with pytest.raises(ClientError) as exc:
        transfer.describe_server(ServerId="s-doesnotexist00000")
    assert exc.value.response["Error"]["Code"] == "ResourceNotFoundException"


def test_transfer_list_servers(transfer):
    resp = transfer.list_servers()
    assert "Servers" in resp
    assert len(resp["Servers"]) >= 1


def test_transfer_create_server_with_options(transfer):
    resp = transfer.create_server(
        EndpointType="VPC",
        Protocols=["SFTP", "FTPS"],
        IdentityProviderType="API_GATEWAY",
        Tags=[{"Key": "env", "Value": "test"}],
    )
    sid = resp["ServerId"]
    server = transfer.describe_server(ServerId=sid)["Server"]
    assert server["EndpointType"] == "VPC"
    assert "FTPS" in server["Protocols"]
    assert server["IdentityProviderType"] == "API_GATEWAY"


def test_transfer_delete_server(transfer):
    sid = transfer.create_server()["ServerId"]
    transfer.delete_server(ServerId=sid)
    with pytest.raises(ClientError) as exc:
        transfer.describe_server(ServerId=sid)
    assert exc.value.response["Error"]["Code"] == "ResourceNotFoundException"


def test_transfer_delete_server_cascades_users(transfer):
    sid = transfer.create_server()["ServerId"]
    transfer.create_user(
        ServerId=sid,
        UserName="cascade-user",
        Role="arn:aws:iam::000000000000:role/transfer-role",
    )
    transfer.delete_server(ServerId=sid)
    # Recreate to verify user is gone
    sid2 = transfer.create_server()["ServerId"]
    resp = transfer.list_users(ServerId=sid2)
    assert len(resp["Users"]) == 0


# ========== User CRUD ==========

@pytest.fixture
def server_id(transfer):
    """Create a fresh server for user tests."""
    return transfer.create_server()["ServerId"]


def test_transfer_create_user(transfer, server_id):
    resp = transfer.create_user(
        ServerId=server_id,
        UserName="test-sftp-user",
        HomeDirectoryType="LOGICAL",
        HomeDirectoryMappings=[{"Entry": "/", "Target": "/my-bucket/path"}],
        Role="arn:aws:iam::000000000000:role/transfer-role",
        SshPublicKeyBody="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQ testkey",
    )
    assert resp["ServerId"] == server_id
    assert resp["UserName"] == "test-sftp-user"


def test_transfer_describe_user(transfer, server_id):
    transfer.create_user(
        ServerId=server_id,
        UserName="describe-user",
        HomeDirectoryType="LOGICAL",
        HomeDirectoryMappings=[{"Entry": "/", "Target": "/bucket/home"}],
        Role="arn:aws:iam::000000000000:role/xfer",
        SshPublicKeyBody="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQ desckey",
    )
    resp = transfer.describe_user(ServerId=server_id, UserName="describe-user")
    user = resp["User"]
    assert user["UserName"] == "describe-user"
    assert user["HomeDirectoryType"] == "LOGICAL"
    assert user["HomeDirectoryMappings"] == [{"Entry": "/", "Target": "/bucket/home"}]
    assert user["Role"] == "arn:aws:iam::000000000000:role/xfer"
    assert len(user["SshPublicKeys"]) == 1
    assert user["SshPublicKeys"][0]["SshPublicKeyBody"] == "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQ desckey"
    assert user["SshPublicKeys"][0]["SshPublicKeyId"].startswith("key-")
    assert user["Arn"].startswith("arn:aws:transfer:")


def test_transfer_create_user_duplicate(transfer, server_id):
    transfer.create_user(
        ServerId=server_id,
        UserName="dup-user",
        Role="arn:aws:iam::000000000000:role/xfer",
    )
    with pytest.raises(ClientError) as exc:
        transfer.create_user(
            ServerId=server_id,
            UserName="dup-user",
            Role="arn:aws:iam::000000000000:role/xfer",
        )
    assert exc.value.response["Error"]["Code"] == "ResourceExistsException"


def test_transfer_create_user_server_not_found(transfer):
    with pytest.raises(ClientError) as exc:
        transfer.create_user(
            ServerId="s-doesnotexist00000",
            UserName="orphan-user",
            Role="arn:aws:iam::000000000000:role/xfer",
        )
    assert exc.value.response["Error"]["Code"] == "ResourceNotFoundException"


def test_transfer_create_user_bad_ssh_key(transfer, server_id):
    with pytest.raises(ClientError) as exc:
        transfer.create_user(
            ServerId=server_id,
            UserName="badkey-user",
            Role="arn:aws:iam::000000000000:role/xfer",
            SshPublicKeyBody="not-a-valid-key",
        )
    assert exc.value.response["Error"]["Code"] == "InvalidRequestException"


def test_transfer_describe_user_not_found(transfer, server_id):
    with pytest.raises(ClientError) as exc:
        transfer.describe_user(ServerId=server_id, UserName="nonexistent")
    assert exc.value.response["Error"]["Code"] == "ResourceNotFoundException"


def test_transfer_delete_user(transfer, server_id):
    transfer.create_user(
        ServerId=server_id,
        UserName="to-delete",
        Role="arn:aws:iam::000000000000:role/xfer",
    )
    transfer.delete_user(ServerId=server_id, UserName="to-delete")
    with pytest.raises(ClientError) as exc:
        transfer.describe_user(ServerId=server_id, UserName="to-delete")
    assert exc.value.response["Error"]["Code"] == "ResourceNotFoundException"


def test_transfer_list_users(transfer, server_id):
    transfer.create_user(
        ServerId=server_id,
        UserName="list-user-a",
        Role="arn:aws:iam::000000000000:role/xfer",
    )
    transfer.create_user(
        ServerId=server_id,
        UserName="list-user-b",
        Role="arn:aws:iam::000000000000:role/xfer",
    )
    resp = transfer.list_users(ServerId=server_id)
    names = [u["UserName"] for u in resp["Users"]]
    assert "list-user-a" in names
    assert "list-user-b" in names


# ========== SSH key management ==========

def test_transfer_import_ssh_key(transfer, server_id):
    transfer.create_user(
        ServerId=server_id,
        UserName="key-user",
        Role="arn:aws:iam::000000000000:role/xfer",
        SshPublicKeyBody="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQ original",
    )
    resp = transfer.import_ssh_public_key(
        ServerId=server_id,
        UserName="key-user",
        SshPublicKeyBody="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIG newkey",
    )
    assert resp["SshPublicKeyId"].startswith("key-")
    assert resp["UserName"] == "key-user"

    user = transfer.describe_user(ServerId=server_id, UserName="key-user")["User"]
    assert len(user["SshPublicKeys"]) == 2
    bodies = {k["SshPublicKeyBody"] for k in user["SshPublicKeys"]}
    assert "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQ original" in bodies
    assert "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIG newkey" in bodies


def test_transfer_import_ssh_key_bad_format(transfer, server_id):
    transfer.create_user(
        ServerId=server_id,
        UserName="badimport-user",
        Role="arn:aws:iam::000000000000:role/xfer",
    )
    with pytest.raises(ClientError) as exc:
        transfer.import_ssh_public_key(
            ServerId=server_id,
            UserName="badimport-user",
            SshPublicKeyBody="invalid-key-format",
        )
    assert exc.value.response["Error"]["Code"] == "InvalidRequestException"


def test_transfer_delete_ssh_key(transfer, server_id):
    transfer.create_user(
        ServerId=server_id,
        UserName="delkey-user",
        Role="arn:aws:iam::000000000000:role/xfer",
        SshPublicKeyBody="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQ first",
    )
    import_resp = transfer.import_ssh_public_key(
        ServerId=server_id,
        UserName="delkey-user",
        SshPublicKeyBody="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQ second",
    )
    new_key_id = import_resp["SshPublicKeyId"]

    # Get the original key ID
    user = transfer.describe_user(ServerId=server_id, UserName="delkey-user")["User"]
    original_key_id = [k["SshPublicKeyId"] for k in user["SshPublicKeys"]
                       if k["SshPublicKeyBody"] == "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQ first"][0]

    # Delete the original key
    transfer.delete_ssh_public_key(
        ServerId=server_id,
        UserName="delkey-user",
        SshPublicKeyId=original_key_id,
    )

    user = transfer.describe_user(ServerId=server_id, UserName="delkey-user")["User"]
    assert len(user["SshPublicKeys"]) == 1
    assert user["SshPublicKeys"][0]["SshPublicKeyId"] == new_key_id
    assert user["SshPublicKeys"][0]["SshPublicKeyBody"] == "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQ second"


# ========== WorkOS end-to-end workflow ==========

def test_transfer_workos_sftp_workflow(transfer):
    """
    Simulates the WorkOS SFTP directory sync workflow:
    1. Server exists (create it)
    2. Create user with LOGICAL home dir + SSH key
    3. Describe user to verify
    4. Rotate SSH key (import new, delete old)
    5. Verify single key remains
    6. Delete user
    """
    # 1. Create server
    sid = transfer.create_server()["ServerId"]

    # 2. Create user with LOGICAL home directory mapping to S3
    transfer.create_user(
        ServerId=sid,
        UserName="sftp-org123",
        HomeDirectoryType="LOGICAL",
        HomeDirectoryMappings=[{"Entry": "/", "Target": "/sftp-org123-bucket/"}],
        Role="arn:aws:iam::000000000000:role/aws_transfer_service_write_only_role",
        SshPublicKeyBody="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQ oldkey",
    )

    # 3. Describe user and verify setup
    user = transfer.describe_user(ServerId=sid, UserName="sftp-org123")["User"]
    assert user["HomeDirectoryType"] == "LOGICAL"
    assert user["HomeDirectoryMappings"] == [{"Entry": "/", "Target": "/sftp-org123-bucket/"}]
    assert len(user["SshPublicKeys"]) == 1
    old_key_id = user["SshPublicKeys"][0]["SshPublicKeyId"]

    # 4. Rotate SSH key: import new key
    import_resp = transfer.import_ssh_public_key(
        ServerId=sid,
        UserName="sftp-org123",
        SshPublicKeyBody="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQ newkey",
    )
    new_key_id = import_resp["SshPublicKeyId"]

    # Verify both keys present
    user = transfer.describe_user(ServerId=sid, UserName="sftp-org123")["User"]
    assert len(user["SshPublicKeys"]) == 2

    # Delete old key
    transfer.delete_ssh_public_key(
        ServerId=sid,
        UserName="sftp-org123",
        SshPublicKeyId=old_key_id,
    )

    # 5. Verify single key remains
    user = transfer.describe_user(ServerId=sid, UserName="sftp-org123")["User"]
    assert len(user["SshPublicKeys"]) == 1
    assert user["SshPublicKeys"][0]["SshPublicKeyId"] == new_key_id
    assert user["SshPublicKeys"][0]["SshPublicKeyBody"] == "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQ newkey"

    # 6. Delete user
    transfer.delete_user(ServerId=sid, UserName="sftp-org123")
    with pytest.raises(ClientError) as exc:
        transfer.describe_user(ServerId=sid, UserName="sftp-org123")
    assert exc.value.response["Error"]["Code"] == "ResourceNotFoundException"
