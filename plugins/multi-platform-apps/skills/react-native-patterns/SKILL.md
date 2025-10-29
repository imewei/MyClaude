# React Native Development Patterns

> **Modern React Native patterns, performance optimization, and cross-platform best practices for production mobile applications.**

---

## Skill Overview

This skill provides comprehensive knowledge for React Native development with the New Architecture, covering component patterns, state management, native integrations, and performance optimization strategies.

**Target Audience**: React/TypeScript developers transitioning to mobile or teams adopting React Native

**Estimated Learning Time**: 5-7 hours to master core concepts

---

## Core Concepts

### 1. Component Architecture

**Key Principle**: Build reusable, performant components with TypeScript for type safety.

#### Basic Component Pattern

```typescript
import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';

interface UserCardProps {
  name: string;
  email: string;
  onPress?: () => void;
}

export const UserCard: React.FC<UserCardProps> = ({ name, email, onPress }) => {
  return (
    <TouchableOpacity style={styles.container} onPress={onPress} activeOpacity={0.7}>
      <View style={styles.avatar}>
        <Text style={styles.avatarText}>{name[0]}</Text>
      </View>
      <View style={styles.content}>
        <Text style={styles.name}>{name}</Text>
        <Text style={styles.email}>{email}</Text>
      </View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    padding: 16,
    backgroundColor: '#fff',
    borderRadius: 8,
    marginVertical: 8,
  },
  avatar: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#4A90E2',
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
  },
  content: {
    marginLeft: 12,
    justifyContent: 'center',
  },
  name: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  email: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
});
```

---

### 2. Performance Optimization

#### Memoization Strategies

```typescript
import React, { memo, useCallback, useMemo } from 'react';
import { FlatList, ListRenderItem } from 'react-native';

interface Item {
  id: string;
  name: string;
  value: number;
}

// ✅ Good: Memoized list item component
const ListItem = memo<{ item: Item; onPress: (id: string) => void }>(
  ({ item, onPress }) => {
    const handlePress = useCallback(() => {
      onPress(item.id);
    }, [item.id, onPress]);

    return (
      <TouchableOpacity onPress={handlePress}>
        <Text>{item.name}: {item.value}</Text>
      </TouchableOpacity>
    );
  }
);

// ✅ Good: Optimized list rendering
export const OptimizedList: React.FC<{ items: Item[] }> = ({ items }) => {
  const handleItemPress = useCallback((id: string) => {
    console.log('Pressed item:', id);
  }, []);

  const renderItem: ListRenderItem<Item> = useCallback(
    ({ item }) => <ListItem item={item} onPress={handleItemPress} />,
    [handleItemPress]
  );

  const keyExtractor = useCallback((item: Item) => item.id, []);

  return (
    <FlatList
      data={items}
      renderItem={renderItem}
      keyExtractor={keyExtractor}
      removeClippedSubviews={true}
      maxToRenderPerBatch={10}
      windowSize={10}
      initialNumToRender={10}
      getItemLayout={(data, index) => ({
        length: ITEM_HEIGHT,
        offset: ITEM_HEIGHT * index,
        index,
      })}
    />
  );
};
```

#### Image Optimization

```typescript
import FastImage from 'react-native-fast-image';

// ✅ Good: Optimized image loading
export const OptimizedImage: React.FC<{
  uri: string;
  width: number;
  height: number;
}> = ({ uri, width, height }) => {
  return (
    <FastImage
      source={{
        uri,
        priority: FastImage.priority.normal,
        cache: FastImage.cacheControl.immutable,
      }}
      style={{ width, height }}
      resizeMode={FastImage.resizeMode.cover}
    />
  );
};

// ✅ Good: Preload images
const preloadImages = async (imageUrls: string[]) => {
  await FastImage.preload(
    imageUrls.map(uri => ({
      uri,
      priority: FastImage.priority.high,
    }))
  );
};
```

---

### 3. State Management with Redux Toolkit

```typescript
// store/slices/userSlice.ts
import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';

interface User {
  id: string;
  name: string;
  email: string;
}

interface UserState {
  user: User | null;
  loading: boolean;
  error: string | null;
}

const initialState: UserState = {
  user: null,
  loading: false,
  error: null,
};

// Async thunk for API calls
export const fetchUser = createAsyncThunk(
  'user/fetchUser',
  async (userId: string, { rejectWithValue }) => {
    try {
      const response = await fetch(`https://api.example.com/users/${userId}`);
      const data = await response.json();
      return data as User;
    } catch (error) {
      return rejectWithValue('Failed to fetch user');
    }
  }
);

const userSlice = createSlice({
  name: 'user',
  initialState,
  reducers: {
    clearUser: (state) => {
      state.user = null;
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchUser.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchUser.fulfilled, (state, action: PayloadAction<User>) => {
        state.loading = false;
        state.user = action.payload;
      })
      .addCase(fetchUser.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });
  },
});

export const { clearUser } = userSlice.actions;
export default userSlice.reducer;

// Usage in component
import { useAppDispatch, useAppSelector } from '../store/hooks';

export const UserProfile: React.FC = () => {
  const dispatch = useAppDispatch();
  const { user, loading, error } = useAppSelector((state) => state.user);

  useEffect(() => {
    dispatch(fetchUser('123'));
  }, [dispatch]);

  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorView message={error} />;
  if (!user) return null;

  return <UserCard name={user.name} email={user.email} />;
};
```

---

### 4. Navigation with React Navigation

```typescript
// navigation/types.ts
export type RootStackParamList = {
  Home: undefined;
  Profile: { userId: string };
  Settings: undefined;
};

// navigation/RootNavigator.tsx
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

const Stack = createNativeStackNavigator<RootStackParamList>();

export const RootNavigator: React.FC = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="Home"
        screenOptions={{
          headerShown: true,
          animation: 'slide_from_right',
        }}
      >
        <Stack.Screen
          name="Home"
          component={HomeScreen}
          options={{ title: 'Welcome' }}
        />
        <Stack.Screen
          name="Profile"
          component={ProfileScreen}
          options={({ route }) => ({ title: `Profile #${route.params.userId}` })}
        />
        <Stack.Screen name="Settings" component={SettingsScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

// Usage in component with TypeScript
import { NativeStackScreenProps } from '@react-navigation/native-stack';

type ProfileScreenProps = NativeStackScreenProps<RootStackParamList, 'Profile'>;

export const ProfileScreen: React.FC<ProfileScreenProps> = ({ route, navigation }) => {
  const { userId } = route.params;

  const navigateToSettings = () => {
    navigation.navigate('Settings');
  };

  return (
    <View>
      <Text>User ID: {userId}</Text>
      <Button title="Go to Settings" onPress={navigateToSettings} />
    </View>
  );
};
```

---

### 5. API Integration with Axios & React Query

```typescript
// api/client.ts
import axios from 'axios';

export const apiClient = axios.create({
  baseURL: 'https://api.example.com',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for auth
apiClient.interceptors.request.use(async (config) => {
  const token = await getAuthToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Add response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
      handleLogout();
    }
    return Promise.reject(error);
  }
);

// hooks/useUser.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

interface User {
  id: string;
  name: string;
  email: string;
}

export const useUser = (userId: string) => {
  return useQuery({
    queryKey: ['user', userId],
    queryFn: async () => {
      const { data } = await apiClient.get<User>(`/users/${userId}`);
      return data;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

export const useUpdateUser = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (user: User) => {
      const { data } = await apiClient.put(`/users/${user.id}`, user);
      return data;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['user', data.id] });
    },
  });
};

// Usage in component
export const UserProfile: React.FC = () => {
  const { data: user, isLoading, error } = useUser('123');
  const updateUser = useUpdateUser();

  const handleUpdate = async () => {
    await updateUser.mutateAsync({
      ...user!,
      name: 'New Name',
    });
  };

  if (isLoading) return <LoadingSpinner />;
  if (error) return <ErrorView message={error.message} />;

  return (
    <View>
      <Text>{user?.name}</Text>
      <Button title="Update" onPress={handleUpdate} />
    </View>
  );
};
```

---

### 6. Form Handling with React Hook Form

```typescript
import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

// Define validation schema
const loginSchema = z.object({
  email: z.string().email('Invalid email format'),
  password: z.string().min(8, 'Password must be at least 8 characters'),
});

type LoginFormData = z.infer<typeof loginSchema>;

export const LoginForm: React.FC = () => {
  const {
    control,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      email: '',
      password: '',
    },
  });

  const onSubmit = async (data: LoginFormData) => {
    try {
      await login(data.email, data.password);
      // Navigate to home
    } catch (error) {
      // Handle error
    }
  };

  return (
    <View style={styles.container}>
      <Controller
        control={control}
        name="email"
        render={({ field: { onChange, onBlur, value } }) => (
          <View>
            <TextInput
              style={styles.input}
              placeholder="Email"
              onBlur={onBlur}
              onChangeText={onChange}
              value={value}
              keyboardType="email-address"
              autoCapitalize="none"
            />
            {errors.email && (
              <Text style={styles.error}>{errors.email.message}</Text>
            )}
          </View>
        )}
      />

      <Controller
        control={control}
        name="password"
        render={({ field: { onChange, onBlur, value } }) => (
          <View>
            <TextInput
              style={styles.input}
              placeholder="Password"
              onBlur={onBlur}
              onChangeText={onChange}
              value={value}
              secureTextEntry
            />
            {errors.password && (
              <Text style={styles.error}>{errors.password.message}</Text>
            )}
          </View>
        )}
      />

      <Button
        title={isSubmitting ? 'Loading...' : 'Login'}
        onPress={handleSubmit(onSubmit)}
        disabled={isSubmitting}
      />
    </View>
  );
};
```

---

### 7. Native Module Integration

```typescript
// Creating a native module bridge
// ios/MyNativeModule.swift
import Foundation
import React

@objc(MyNativeModule)
class MyNativeModule: NSObject {

  @objc
  func getBatteryLevel(_ resolve: @escaping RCTPromiseResolveBlock,
                       reject: @escaping RCTPromiseRejectBlock) {
    UIDevice.current.isBatteryMonitoringEnabled = true
    let batteryLevel = UIDevice.current.batteryLevel

    if batteryLevel < 0 {
      reject("ERROR", "Unable to get battery level", nil)
    } else {
      resolve(batteryLevel * 100)
    }
  }

  @objc
  static func requiresMainQueueSetup() -> Bool {
    return false
  }
}

// TypeScript interface
// NativeModules.ts
import { NativeModules } from 'react-native';

interface MyNativeModule {
  getBatteryLevel(): Promise<number>;
}

export const { MyNativeModule } = NativeModules as {
  MyNativeModule: MyNativeModule;
};

// Usage
export const BatteryLevel: React.FC = () => {
  const [batteryLevel, setBatteryLevel] = useState<number | null>(null);

  useEffect(() => {
    MyNativeModule.getBatteryLevel()
      .then(setBatteryLevel)
      .catch(console.error);
  }, []);

  return (
    <View>
      <Text>Battery: {batteryLevel?.toFixed(0)}%</Text>
    </View>
  );
};
```

---

### 8. Offline Support with Async Storage

```typescript
import AsyncStorage from '@react-native-async-storage/async-storage';

// Storage utilities
export const storage = {
  async set<T>(key: string, value: T): Promise<void> {
    await AsyncStorage.setItem(key, JSON.stringify(value));
  },

  async get<T>(key: string): Promise<T | null> {
    const value = await AsyncStorage.getItem(key);
    return value ? JSON.parse(value) : null;
  },

  async remove(key: string): Promise<void> {
    await AsyncStorage.removeItem(key);
  },

  async clear(): Promise<void> {
    await AsyncStorage.clear();
  },
};

// Offline-first data hook
export const useOfflineFirst = <T,>(
  key: string,
  fetchFn: () => Promise<T>
) => {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      // Load from cache first
      const cached = await storage.get<T>(key);
      if (cached) {
        setData(cached);
        setLoading(false);
      }

      // Fetch fresh data
      try {
        const fresh = await fetchFn();
        setData(fresh);
        await storage.set(key, fresh);
      } catch (error) {
        console.error('Failed to fetch:', error);
        // Use cached data if fetch fails
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [key]);

  return { data, loading };
};
```

---

## Architecture Patterns

### Feature-Based Structure

```
src/
├── features/
│   ├── auth/
│   │   ├── components/
│   │   │   ├── LoginForm.tsx
│   │   │   └── SignupForm.tsx
│   │   ├── screens/
│   │   │   ├── LoginScreen.tsx
│   │   │   └── SignupScreen.tsx
│   │   ├── hooks/
│   │   │   └── useAuth.ts
│   │   └── api/
│   │       └── authApi.ts
│   ├── user/
│   │   ├── components/
│   │   ├── screens/
│   │   ├── hooks/
│   │   └── api/
│   └── posts/
│       ├── components/
│       ├── screens/
│       ├── hooks/
│       └── api/
├── shared/
│   ├── components/
│   ├── hooks/
│   ├── utils/
│   └── types/
├── navigation/
├── store/
└── App.tsx
```

---

## Quick Reference

### Essential Packages

| Package | Purpose | Use Case |
|---------|---------|----------|
| `@tanstack/react-query` | Server state | API data fetching |
| `@reduxjs/toolkit` | Client state | Complex app state |
| `react-navigation` | Navigation | Screen transitions |
| `react-hook-form` | Forms | Form validation |
| `axios` | HTTP client | API requests |
| `react-native-fast-image` | Images | Image optimization |
| `@react-native-async-storage` | Storage | Local persistence |

### Performance Checklist

- [ ] Use `React.memo` for expensive components
- [ ] Use `useCallback` for function props
- [ ] Use `useMemo` for expensive computations
- [ ] Optimize FlatList with proper props
- [ ] Use Fast Image for network images
- [ ] Implement lazy loading for screens
- [ ] Profile with Flipper before optimizing
- [ ] Use Hermes JavaScript engine

---

## Anti-Patterns to Avoid

### ❌ Don't: Inline styles and functions

```typescript
// Bad: Creates new objects on every render
<View style={{ padding: 16, margin: 8 }}>
  <Button onPress={() => console.log('press')} />
</View>
```

### ✅ Do: Extract styles and memoize callbacks

```typescript
// Good: Reuses style object and memoized function
const handlePress = useCallback(() => {
  console.log('press');
}, []);

<View style={styles.container}>
  <Button onPress={handlePress} />
</View>

const styles = StyleSheet.create({
  container: { padding: 16, margin: 8 },
});
```

---

**Skill Version**: 1.0.0
**Last Updated**: October 27, 2024
**Difficulty**: Intermediate
**Estimated Time**: 5-7 hours
