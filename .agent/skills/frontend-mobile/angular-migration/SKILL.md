---
name: angular-migration
version: "1.0.7"
maturity: "5-Expert"
specialization: AngularJS to Angular
description: Migrate AngularJS (1.x) to Angular (2+) using hybrid mode, ngUpgrade, component conversion, dependency injection updates, and routing migration. Use when upgrading legacy AngularJS apps incrementally or implementing hybrid AngularJS/Angular applications.
---

# Angular Migration

AngularJS to Angular migration with hybrid apps and incremental conversion.

---

## Migration Strategies

| Strategy | Code Sharing | Best For | Risk |
|----------|--------------|----------|------|
| Big Bang | None (rewrite) | Small apps, greenfield | High |
| Hybrid (ngUpgrade) | Side-by-side | Large apps, continuous delivery | Low |
| Vertical Slice | Feature-complete | Medium apps, distinct features | Medium |

---

## Hybrid App Setup

```typescript
// main.ts - Bootstrap hybrid app
import { platformBrowserDynamic } from '@angular/platform-browser-dynamic';
import { UpgradeModule } from '@angular/upgrade/static';
import { AppModule } from './app/app.module';

platformBrowserDynamic()
  .bootstrapModule(AppModule)
  .then(platformRef => {
    const upgrade = platformRef.injector.get(UpgradeModule);
    upgrade.bootstrap(document.body, ['myAngularJSApp'], { strictDi: true });
  });
```

```typescript
// app.module.ts
@NgModule({
  imports: [BrowserModule, UpgradeModule]
})
export class AppModule {
  constructor(private upgrade: UpgradeModule) {}
  ngDoBootstrap() {} // Bootstrapped manually in main.ts
}
```

---

## Component Conversion

### Controller → Component

```javascript
// Before: AngularJS
angular.module('myApp').controller('UserController', function($scope, UserService) {
  $scope.user = {};
  $scope.loadUser = function(id) {
    UserService.getUser(id).then(user => $scope.user = user);
  };
});
```

```typescript
// After: Angular
@Component({
  selector: 'app-user',
  template: `<h2>{{ user.name }}</h2><button (click)="saveUser()">Save</button>`
})
export class UserComponent implements OnInit {
  user: any = {};
  constructor(private userService: UserService) {}

  ngOnInit() { this.loadUser(1); }

  loadUser(id: number) {
    this.userService.getUser(id).subscribe(user => this.user = user);
  }
}
```

### Directive → Component

```javascript
// Before: AngularJS directive
angular.module('myApp').directive('userCard', function() {
  return {
    restrict: 'E',
    scope: { user: '=', onDelete: '&' },
    template: `<div><h3>{{ user.name }}</h3><button ng-click="onDelete()">Delete</button></div>`
  };
});
```

```typescript
// After: Angular component
@Component({
  selector: 'app-user-card',
  template: `<div><h3>{{ user.name }}</h3><button (click)="delete.emit()">Delete</button></div>`
})
export class UserCardComponent {
  @Input() user: any;
  @Output() delete = new EventEmitter<void>();
}
```

---

## Service Migration

```javascript
// Before: AngularJS factory
angular.module('myApp').factory('UserService', function($http) {
  return {
    getUser: id => $http.get('/api/users/' + id),
    saveUser: user => $http.post('/api/users', user)
  };
});
```

```typescript
// After: Angular service
@Injectable({ providedIn: 'root' })
export class UserService {
  constructor(private http: HttpClient) {}

  getUser(id: number): Observable<any> {
    return this.http.get(`/api/users/${id}`);
  }

  saveUser(user: any): Observable<any> {
    return this.http.post('/api/users', user);
  }
}
```

---

## Interoperability

### Downgrade Angular → AngularJS

```typescript
import { downgradeComponent, downgradeInjectable } from '@angular/upgrade/static';

// Use Angular component in AngularJS
angular.module('myApp')
  .directive('appUser', downgradeComponent({ component: UserComponent }));

// Use Angular service in AngularJS
angular.module('myApp')
  .factory('userService', downgradeInjectable(UserService));
```

### Upgrade AngularJS → Angular

```typescript
export const OLD_SERVICE = new InjectionToken<any>('oldService');

@NgModule({
  providers: [{
    provide: OLD_SERVICE,
    useFactory: (i: any) => i.get('oldService'),
    deps: ['$injector']
  }]
})

// Use in Angular component
@Component({...})
export class NewComponent {
  constructor(@Inject(OLD_SERVICE) private oldService: any) {}
}
```

---

## Routing Migration

```javascript
// Before: AngularJS
$routeProvider
  .when('/users', { template: '<user-list></user-list>' })
  .when('/users/:id', { template: '<user-detail></user-detail>' });
```

```typescript
// After: Angular
const routes: Routes = [
  { path: 'users', component: UserListComponent },
  { path: 'users/:id', component: UserDetailComponent }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule {}
```

---

## Forms Migration

```html
<!-- Before: AngularJS -->
<form ng-submit="saveUser()">
  <input ng-model="user.name" required>
  <button ng-disabled="userForm.$invalid">Save</button>
</form>
```

```typescript
// After: Angular Reactive Forms
@Component({
  template: `
    <form [formGroup]="userForm" (ngSubmit)="saveUser()">
      <input formControlName="name">
      <button [disabled]="userForm.invalid">Save</button>
    </form>`
})
export class UserFormComponent {
  userForm = this.fb.group({ name: ['', Validators.required] });
  constructor(private fb: FormBuilder) {}
}
```

---

## Migration Order

| Phase | Focus | Duration |
|-------|-------|----------|
| Setup | Hybrid app, build tools | 1-2 weeks |
| Infrastructure | Services, utilities | 2-4 weeks |
| Features | Components, feature-by-feature | Varies |
| Cleanup | Remove AngularJS, optimize | 1-2 weeks |

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Services first | Easiest to migrate |
| Incremental | Feature-by-feature |
| Test continuously | After each migration |
| TypeScript early | Migrate to TS first |
| Angular style guide | From day 1 |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Wrong hybrid setup | Follow bootstrap exactly |
| UI before logic | Migrate services first |
| Change detection issues | Understand zone differences |
| $scope reliance | Use component properties |
| Mixing patterns | Keep clear boundaries |

---

## Checklist

- [ ] Hybrid app bootstrap working
- [ ] UpgradeModule configured
- [ ] Services migrated and injectable
- [ ] Components converted with @Input/@Output
- [ ] Routing migrated to Angular Router
- [ ] Forms converted (reactive preferred)
- [ ] AngularJS code removed
- [ ] Bundle optimized

---

**Version**: 1.0.5
