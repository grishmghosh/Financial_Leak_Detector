from fastapi import Depends
from app.core.tenant import apply_tenant_context

TenantDependency = Depends(apply_tenant_context)
