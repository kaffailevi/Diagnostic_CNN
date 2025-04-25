import {CanActivateFn, Router} from '@angular/router';
import {inject} from '@angular/core';
import {AuthService} from '../service/auth.service';

export const authGuard: CanActivateFn = (route, state) => {
  const authService = inject(AuthService);
  const router = inject(Router);
  return authService.isAuthenticated().then(isAuthenticated => {
    if (!isAuthenticated) {
      router.navigate(['']);
    }
    return isAuthenticated;
  });
};
