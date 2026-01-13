---
name: database-migration
version: "1.0.7"
maturity: "5-Expert"
specialization: Schema Evolution
description: Execute database migrations across ORMs (Sequelize, TypeORM, Prisma, Django) with zero-downtime strategies, rollback procedures, and data transformations. Use when adding/removing columns, renaming tables, changing types, or implementing blue-green deployments.
---

# Database Migration

Safe schema evolution with rollback strategies and zero-downtime patterns.

---

## ORM Migration Commands

| ORM | Create Migration | Run | Rollback |
|-----|-----------------|-----|----------|
| Sequelize | `sequelize migration:create` | `sequelize db:migrate` | `db:migrate:undo` |
| TypeORM | `typeorm migration:create` | `migration:run` | `migration:revert` |
| Prisma | `prisma migrate dev` | `prisma migrate deploy` | N/A (manual) |
| Django | `makemigrations` | `migrate` | `migrate app 0001` |

---

## Migration Patterns

### Sequelize Example

```javascript
module.exports = {
  up: async (queryInterface, Sequelize) => {
    await queryInterface.addColumn('users', 'status', {
      type: Sequelize.STRING,
      defaultValue: 'active',
      allowNull: false
    });
  },
  down: async (queryInterface) => {
    await queryInterface.removeColumn('users', 'status');
  }
};
```

### TypeORM Example

```typescript
export class AddUserStatus implements MigrationInterface {
  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.addColumn('users', new TableColumn({
      name: 'status', type: 'varchar', default: "'active'"
    }));
  }
  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.dropColumn('users', 'status');
  }
}
```

---

## Zero-Downtime Column Rename

Multi-phase approach for backwards compatibility:

| Phase | Action | Code State |
|-------|--------|------------|
| 1 | Add new column | Reads old, writes old |
| 2 | Backfill data | Reads old, writes both |
| 3 | Deploy new code | Reads new, writes both |
| 4 | Drop old column | Reads new, writes new |

```javascript
// Phase 1: Add new column
module.exports = {
  up: async (queryInterface, Sequelize) => {
    await queryInterface.addColumn('users', 'full_name', { type: Sequelize.STRING });
    await queryInterface.sequelize.query('UPDATE users SET full_name = name');
  },
  down: async (queryInterface) => {
    await queryInterface.removeColumn('users', 'full_name');
  }
};

// Phase 4: Remove old column (after code deployment)
module.exports = {
  up: async (queryInterface) => {
    await queryInterface.removeColumn('users', 'name');
  },
  down: async (queryInterface, Sequelize) => {
    await queryInterface.addColumn('users', 'name', { type: Sequelize.STRING });
  }
};
```

---

## Transaction-Based Migration

```javascript
module.exports = {
  up: async (queryInterface, Sequelize) => {
    const transaction = await queryInterface.sequelize.transaction();
    try {
      await queryInterface.addColumn('users', 'verified',
        { type: Sequelize.BOOLEAN, defaultValue: false }, { transaction });
      await queryInterface.sequelize.query(
        'UPDATE users SET verified = true WHERE email_verified_at IS NOT NULL',
        { transaction });
      await transaction.commit();
    } catch (error) {
      await transaction.rollback();
      throw error;
    }
  },
  down: async (queryInterface) => {
    await queryInterface.removeColumn('users', 'verified');
  }
};
```

---

## Data Transformation

```javascript
module.exports = {
  up: async (queryInterface) => {
    const [users] = await queryInterface.sequelize.query(
      'SELECT id, address_string FROM users');

    for (const user of users) {
      const [street, city, state] = user.address_string.split(',').map(s => s.trim());
      await queryInterface.sequelize.query(
        `UPDATE users SET street = :street, city = :city, state = :state WHERE id = :id`,
        { replacements: { id: user.id, street, city, state } });
    }
    await queryInterface.removeColumn('users', 'address_string');
  }
};
```

---

## Cross-Database Handling

```javascript
module.exports = {
  up: async (queryInterface, Sequelize) => {
    const dialect = queryInterface.sequelize.getDialect();
    const jsonType = dialect === 'postgres' ? Sequelize.JSONB : Sequelize.JSON;

    await queryInterface.createTable('configs', {
      id: { type: Sequelize.INTEGER, primaryKey: true, autoIncrement: true },
      data: { type: jsonType }
    });
  }
};
```

---

## Rollback Strategies

| Strategy | When to Use |
|----------|-------------|
| Transaction-based | Atomic changes, PostgreSQL |
| Checkpoint backup | Critical data, large tables |
| Feature flags | Gradual rollout |
| Blue-green | Zero-downtime required |

### Checkpoint Backup

```javascript
module.exports = {
  up: async (queryInterface) => {
    await queryInterface.sequelize.query('CREATE TABLE users_backup AS SELECT * FROM users');
    try {
      // Migration operations...
      await queryInterface.dropTable('users_backup');
    } catch (error) {
      await queryInterface.sequelize.query('DROP TABLE users; ALTER TABLE users_backup RENAME TO users');
      throw error;
    }
  }
};
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Always provide down() | Every up needs a down |
| Test on staging | Mirror production data |
| Use transactions | Atomic when possible |
| Backup first | Before any migration |
| Small increments | One change per migration |
| Idempotent | Safe to re-run |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Breaking changes without downtime plan | Use multi-phase migration |
| No rollback testing | Test down() in staging |
| NULL handling forgotten | Set defaults or migrate data |
| Large data migration | Batch in chunks |
| Foreign key conflicts | Order migrations correctly |

---

## Checklist

- [ ] Rollback procedure tested
- [ ] Staging migration verified
- [ ] Backup created before production
- [ ] Zero-downtime strategy if needed
- [ ] Data transformation validated
- [ ] Foreign key dependencies handled

---

**Version**: 1.0.5
