---
name: react-native-patterns
version: "1.0.7"
maturity: "5-Expert"
specialization: React Native Development
description: Modern React Native patterns with New Architecture, TypeScript, and performance optimization. Use when building React Native components, optimizing FlatList performance, implementing state management (Redux Toolkit, Zustand), setting up React Navigation, integrating APIs with React Query, handling forms with React Hook Form, creating native modules, or implementing offline-first storage.
---

# React Native Patterns

Production React Native with modern architecture and TypeScript.

---

<!-- SECTION: ARCHITECTURE -->
## Architecture

| Complexity | State | Navigation |
|------------|-------|------------|
| Simple | Context + useState | Stack |
| Medium | Zustand | Tab + Stack |
| Complex | Redux Toolkit | Nested navigators |
| Enterprise | Redux + RTK Query | Deep linking + tabs |
<!-- END_SECTION: ARCHITECTURE -->

---

<!-- SECTION: COMPONENT -->
## Component Pattern

```typescript
interface UserCardProps {
  name: string;
  email: string;
  onPress?: () => void;
}

export const UserCard: React.FC<UserCardProps> = ({ name, email, onPress }) => (
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

const styles = StyleSheet.create({
  container: { flexDirection: 'row', padding: 16, backgroundColor: '#fff' },
  avatar: { width: 48, height: 48, borderRadius: 24, backgroundColor: '#4A90E2' },
  // ...
});
```
<!-- END_SECTION: COMPONENT -->

---

<!-- SECTION: PERFORMANCE -->
## Performance

```typescript
// Memoized list item
const ListItem = memo<{ item: Item; onPress: (id: string) => void }>(
  ({ item, onPress }) => {
    const handlePress = useCallback(() => onPress(item.id), [item.id, onPress]);
    return (
      <TouchableOpacity onPress={handlePress}>
        <Text>{item.name}</Text>
      </TouchableOpacity>
    );
  }
);

// Optimized FlatList
export const OptimizedList: React.FC<{ items: Item[] }> = ({ items }) => {
  const handlePress = useCallback((id: string) => console.log(id), []);

  const renderItem = useCallback<ListRenderItem<Item>>(
    ({ item }) => <ListItem item={item} onPress={handlePress} />,
    [handlePress]
  );

  return (
    <FlatList
      data={items}
      renderItem={renderItem}
      keyExtractor={(item) => item.id}
      removeClippedSubviews={true}
      maxToRenderPerBatch={10}
      windowSize={10}
      getItemLayout={(_, index) => ({ length: 60, offset: 60 * index, index })}
    />
  );
};

// Image optimization with FastImage
import FastImage from 'react-native-fast-image';

<FastImage
  source={{ uri, priority: FastImage.priority.normal, cache: FastImage.cacheControl.immutable }}
  style={{ width: 100, height: 100 }}
  resizeMode={FastImage.resizeMode.cover}
/>
```
<!-- END_SECTION: PERFORMANCE -->

---

<!-- SECTION: REDUX -->
## Redux Toolkit

```typescript
// userSlice.ts
interface UserState {
  user: User | null;
  loading: boolean;
  error: string | null;
}

export const fetchUser = createAsyncThunk(
  'user/fetchUser',
  async (userId: string, { rejectWithValue }) => {
    try {
      const response = await fetch(`/api/users/${userId}`);
      return await response.json();
    } catch (error) {
      return rejectWithValue('Failed to fetch user');
    }
  }
);

const userSlice = createSlice({
  name: 'user',
  initialState: { user: null, loading: false, error: null },
  reducers: {
    clearUser: (state) => { state.user = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchUser.pending, (state) => { state.loading = true; })
      .addCase(fetchUser.fulfilled, (state, action) => {
        state.loading = false;
        state.user = action.payload;
      })
      .addCase(fetchUser.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });
  },
});
```
<!-- END_SECTION: REDUX -->

---

<!-- SECTION: NAVIGATION -->
## Navigation

```typescript
// Type-safe navigation with React Navigation
export type RootStackParamList = {
  Home: undefined;
  Profile: { userId: string };
  Settings: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export const RootNavigator: React.FC = () => (
  <NavigationContainer>
    <Stack.Navigator initialRouteName="Home" screenOptions={{ animation: 'slide_from_right' }}>
      <Stack.Screen name="Home" component={HomeScreen} />
      <Stack.Screen
        name="Profile"
        component={ProfileScreen}
        options={({ route }) => ({ title: `Profile #${route.params.userId}` })}
      />
    </Stack.Navigator>
  </NavigationContainer>
);

type ProfileScreenProps = NativeStackScreenProps<RootStackParamList, 'Profile'>;

export const ProfileScreen: React.FC<ProfileScreenProps> = ({ route, navigation }) => {
  const { userId } = route.params;
  return <Button title="Settings" onPress={() => navigation.navigate('Settings')} />;
};
```
<!-- END_SECTION: NAVIGATION -->

---

<!-- SECTION: API -->
## API (React Query)

```typescript
// api/client.ts
export const apiClient = axios.create({
  baseURL: 'https://api.example.com',
  timeout: 10000,
});

apiClient.interceptors.request.use(async (config) => {
  const token = await getAuthToken();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// hooks/useUser.ts
export const useUser = (userId: string) => {
  return useQuery({
    queryKey: ['user', userId],
    queryFn: async () => {
      const { data } = await apiClient.get<User>(`/users/${userId}`);
      return data;
    },
    staleTime: 5 * 60 * 1000,
  });
};

export const useUpdateUser = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (user: User) => apiClient.put(`/users/${user.id}`, user),
    onSuccess: (_, user) => queryClient.invalidateQueries({ queryKey: ['user', user.id] }),
  });
};
```
<!-- END_SECTION: API -->

---

<!-- SECTION: FORMS -->
## Forms (React Hook Form + Zod)

```typescript
const loginSchema = z.object({
  email: z.string().email('Invalid email'),
  password: z.string().min(8, 'Min 8 characters'),
});

type LoginFormData = z.infer<typeof loginSchema>;

export const LoginForm: React.FC = () => {
  const { control, handleSubmit, formState: { errors, isSubmitting } } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
  });

  return (
    <View>
      <Controller
        control={control}
        name="email"
        render={({ field: { onChange, onBlur, value } }) => (
          <>
            <TextInput placeholder="Email" onBlur={onBlur} onChangeText={onChange} value={value} />
            {errors.email && <Text style={styles.error}>{errors.email.message}</Text>}
          </>
        )}
      />
      <Button title={isSubmitting ? 'Loading...' : 'Login'} onPress={handleSubmit(onSubmit)} />
    </View>
  );
};
```
<!-- END_SECTION: FORMS -->

---

<!-- SECTION: OFFLINE -->
## Offline Storage

```typescript
import AsyncStorage from '@react-native-async-storage/async-storage';

export const storage = {
  async set<T>(key: string, value: T): Promise<void> {
    await AsyncStorage.setItem(key, JSON.stringify(value));
  },
  async get<T>(key: string): Promise<T | null> {
    const value = await AsyncStorage.getItem(key);
    return value ? JSON.parse(value) : null;
  },
};

// Offline-first hook
export const useOfflineFirst = <T,>(key: string, fetchFn: () => Promise<T>) => {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      const cached = await storage.get<T>(key);
      if (cached) { setData(cached); setLoading(false); }

      try {
        const fresh = await fetchFn();
        setData(fresh);
        await storage.set(key, fresh);
      } finally { setLoading(false); }
    })();
  }, [key]);

  return { data, loading };
};
```
<!-- END_SECTION: OFFLINE -->

---

<!-- SECTION: STRUCTURE -->
## Project Structure

```
src/
├── features/
│   ├── auth/
│   │   ├── components/
│   │   ├── screens/
│   │   ├── hooks/
│   │   └── api/
│   └── user/
├── shared/
│   ├── components/
│   ├── hooks/
│   └── utils/
├── navigation/
├── store/
└── App.tsx
```
<!-- END_SECTION: STRUCTURE -->

---

## Packages

| Package | Purpose |
|---------|---------|
| @tanstack/react-query | Server state, caching |
| @reduxjs/toolkit | Client state |
| @react-navigation/native | Navigation |
| react-hook-form | Form handling |
| zod | Validation |
| react-native-fast-image | Image optimization |
| @react-native-async-storage | Local storage |

---

## Checklist

- [ ] Use `React.memo` for expensive components
- [ ] Use `useCallback` for function props
- [ ] Optimize FlatList with `getItemLayout`, `windowSize`
- [ ] Use FastImage for network images
- [ ] Extract styles to `StyleSheet.create`
- [ ] Implement type-safe navigation
- [ ] Use React Query for server state
- [ ] Enable Hermes JavaScript engine
- [ ] Profile with Flipper before optimizing

---

**Version**: 1.0.7
